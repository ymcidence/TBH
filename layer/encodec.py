from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer.binary_activation import binary_activation


class Encoder(tf.keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """

        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(Encoder, self).__init__()
        self.code_length = bbn_dim
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu')
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim)
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim, activation='sigmoid')

    def call(self, inputs, training=True, **kwargs):
        batch_size = tf.shape(inputs)[0]
        fc_1 = self.fc_1(inputs)
        bbn = self.fc_2_1(fc_1)
        # eps = tf.cond(training, lambda: tf.random.uniform([batch_size, self.code_length], maxval=1),
        #               lambda: tf.ones([batch_size, self.code_length]) / 2.)
        eps = tf.ones([batch_size, self.code_length]) / 2.

        bbn, _ = binary_activation(bbn, eps)
        cbn = self.fc_2_2(fc_1)
        return bbn, cbn


# noinspection PyAbstractClass
class Decoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim, feat_dim):
        """
        :param middle_dim: hidden units
        :param feat_dim: data dim
        """
        super(Decoder, self).__init__()
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(feat_dim, activation='relu')

    def call(self, inputs, **kwargs):
        fc_1 = self.fc_1(inputs)
        return self.fc_2(fc_1)


if __name__ == '__main__':
    a = tf.ones([2, 4096], dtype=tf.float32)
    encoder = Encoder(1024, 64, 512)
    b = encoder(a)
    print(encoder.trainable_variables)
