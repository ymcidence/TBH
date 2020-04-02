from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model.jmlh import JMLH
from util.data.dataset import Dataset
from util.eval_tools import eval_cls_map
from meta import REPO_PATH
import os
from time import gmtime, strftime


@tf.function
def kld_loss(q: tf.Tensor, p=0.5):
    loss = q * tf.math.log(q) - q * tf.math.log(p) + (-q + 1) * tf.math.log(-q + 1) - (-q + 1) * tf.math.log(1 - p)
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))


@tf.function
def jmlh_loss(prob, cls_prob, label):
    kld = kld_loss(prob)
    cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, cls_prob))
    return kld * .2 + cls


def train_step(model: JMLH, batch_data, opt: tf.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        model_input = batch_data
        code, prob, cls_prob = model(model_input, training=True)

        loss = jmlh_loss(prob, cls_prob, label=batch_data[2])

        gradient = tape.gradient(loss, sources=model.trainable_variables)
        opt.apply_gradients(zip(gradient, model.trainable_variables))

    return code.numpy(), loss.numpy()


def test_step(model: JMLH, batch_data):
    model_input = batch_data[1]
    model_output = model.run(model_input)
    return model_output.numpy()


def train(set_name, bbn_dim, batch_size, max_iter=100000):
    model = JMLH(set_name, bbn_dim)

    data = Dataset(set_name=set_name, batch_size=batch_size)

    opt = tf.keras.optimizers.Adam(1e-4)

    train_iter = iter(data.train_data)
    test_iter = iter(data.test_data)

    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    result_path = os.path.join(REPO_PATH, 'result', set_name + '_JMLH')
    save_path = os.path.join(result_path, 'model', time_string)
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(opt=opt, model=model)
    for i in range(max_iter):
        with writer.as_default():
            train_batch = next(train_iter)
            train_code, loss = train_step(model, train_batch, opt)
            train_label = train_batch[2].numpy()
            train_entry = train_batch[0].numpy()
            data.update(train_entry, train_code, train_label, 'train')

            if i == 0:
                print(model.summary())

            if (i + 1) % 100 == 0:
                train_hook = eval_cls_map(train_code, train_code, train_label, train_label)

                tf.summary.scalar('train/loss', loss, step=i)
                tf.summary.scalar('train/hook', train_hook, step=i)

                print('batch {}: loss {}'.format(i, loss))

            if (i + 1) % 2000 == 0:
                print('Testing!!!!!!!!')
                test_batch = next(test_iter)
                test_code = test_step(model, test_batch)
                test_label = test_batch[2].numpy()
                test_entry = test_batch[0].numpy()
                data.update(test_entry, test_code, test_label, 'test')
                test_hook = eval_cls_map(test_code, data.train_code, test_label, data.train_label, at=1000)
                tf.summary.scalar('test/hook', test_hook, step=i)

                save_name = os.path.join(save_path, 'ymmodel' + str(i))
                checkpoint.save(file_prefix=save_name)
