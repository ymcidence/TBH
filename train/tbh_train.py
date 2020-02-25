from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from model.tbh import TBH
from util.data.dataset import Dataset


def critic_loss(real, fake):
    real_loss = tf.keras.BinaryCrossentropy(tf.ones_like(real), real)
    fake_loss = tf.keras.BinaryCrossentropy(tf.zeros_like(fake), fake)
    total_loss = real_loss + fake_loss
    return total_loss


def reconstruction_loss(pred, origin):
    return tf.reduce_mean(tf.nn.l2_loss(pred - origin))


@tf.function
def train_step(model, batch_data, bbn_dim, cbn_dim, batch_size):
    random_binary = (tf.sign(tf.random.uniform([batch_size, bbn_dim]) - 0.5) + 1) / 2
    random_cont = tf.random.uniform([batch_size, cbn_dim])


def train(set_name, bbn_dim, cbn_dim, batch_size, middle_dim=1024, max_iter=100000):
    model = TBH(set_name, bbn_dim, cbn_dim, middle_dim)
    data = Dataset(set_name=set_name, batch_size=batch_size)

    train_iter = iter(data.train_data)
    test_iter = iter(data.test_data)
    for i in range(max_iter):
        train_batch = next(train_iter)
