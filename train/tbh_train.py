from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model.tbh import TBH
from util.data.dataset import Dataset
from util.eval_tools import eval_cls_map
from meta import REPO_PATH
import os
from time import gmtime, strftime


def hook(query, base, label_q, label_b):
    return eval_cls_map(query, base, label_q, label_b)


@tf.function
def adv_loss(real, fake):
    real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real), real))
    fake_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake), fake))
    total_loss = real_loss + fake_loss
    return total_loss


@tf.function
def reconstruction_loss(pred, origin):
    return tf.reduce_mean(tf.nn.l2_loss(pred - origin))


def train_step(model: TBH, batch_data, bbn_dim, cbn_dim, batch_size, actor_opt: tf.optimizers.Optimizer,
               critic_opt: tf.optimizers.Optimizer):
    random_binary = (tf.sign(tf.random.uniform([batch_size, bbn_dim]) - 0.5) + 1) / 2
    random_cont = tf.random.uniform([batch_size, cbn_dim])

    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        model_input = [batch_data, random_binary, random_cont]
        model_output = model(model_input, training=True)

        actor_loss = reconstruction_loss(model_output[1], batch_data[1]) - \
                     adv_loss(model_output[4], model_output[2]) - \
                     adv_loss(model_output[5], model_output[3])

        critic_loss = adv_loss(model_output[4], model_output[2]) + adv_loss(model_output[5], model_output[3])

        actor_scope = model.encoder.trainable_variables + model.tbn.trainable_variables + \
                      model.decoder.trainable_variables
        critic_scope = model.dis_1.trainable_variables + model.dis_2.trainable_variables

        actor_gradient = actor_tape.gradient(actor_loss, sources=actor_scope)
        critic_gradient = critic_tape.gradient(critic_loss, sources=critic_scope)

        actor_opt.apply_gradients(zip(actor_gradient, actor_scope))
        critic_opt.apply_gradients(zip(critic_gradient, critic_scope))

    return model_output[0].numpy(), actor_loss.numpy(), critic_loss.numpy()


def test_step(model: TBH, batch_data):
    model_input = [batch_data]
    model_output = model(model_input, training=False)
    return model_output.numpy()


def train(set_name, bbn_dim, cbn_dim, batch_size, middle_dim=1024, max_iter=100000):
    model = TBH(set_name, bbn_dim, cbn_dim, middle_dim)

    data = Dataset(set_name=set_name, batch_size=batch_size)

    actor_opt = tf.keras.optimizers.Adam(1e-4)
    critic_opt = tf.keras.optimizers.Adam(1e-4)

    train_iter = iter(data.train_data)
    test_iter = iter(data.test_data)

    time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
    result_path = os.path.join(REPO_PATH, 'result', set_name)
    save_path = os.path.join(result_path, 'model', time_string)
    summary_path = os.path.join(result_path, 'log', time_string)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = tf.summary.create_file_writer(summary_path)
    checkpoint = tf.train.Checkpoint(actor_opt=actor_opt, critic_opt=critic_opt, model=model)

    for i in range(max_iter):
        with writer.as_default():
            train_batch = next(train_iter)
            train_code, actor_loss, critic_loss = train_step(model, train_batch, bbn_dim, cbn_dim, batch_size,
                                                             actor_opt,
                                                             critic_opt)
            train_label = train_batch[2].numpy()
            train_entry = train_batch[0].numpy()
            data.update(train_entry, train_code, train_label, 'train')

            if i == 0:
                print(model.summary())

            if (i + 1) % 100 == 0:
                train_hook = hook(train_code, train_code, train_label, train_label)

                tf.summary.scalar('train/actor', actor_loss, step=i)
                tf.summary.scalar('train/critic', critic_loss, step=i)
                tf.summary.scalar('train/hook', train_hook, step=i)

                print('batch {}: actor {}, critic {}'.format(i, actor_loss, critic_loss))

            if (i + 1) % 2000 == 0:
                print('Testing!!!!!!!!')
                test_batch = next(test_iter)
                test_code = test_step(model, test_batch)
                test_label = test_batch[2].numpy()
                test_entry = test_batch[0].numpy()
                data.update(test_entry, test_code, test_label, 'test')
                test_hook = eval_cls_map(test_code, data.train_code, test_label, data.train_label, at=1000)
                tf.summary.scalar('test/hook', test_hook, step=i)

                save_name = os.path.join(save_path, 'ymmodel' + str(i) )
                checkpoint.save(file_prefix=save_name)


if __name__ == '__main__':
    train('cifar10', 32, 512, 400)
