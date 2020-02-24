from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim, bbn_dim, cbn_dim):
        """

        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(Encoder, self).__init__()
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu')
        self.fc_2_1 = tf.keras.layers.Dense(bbn_dim)
        self.fc_2_2 = tf.keras.layers.Dense(cbn_dim, activation='sigmoid')

    def call(self, inputs, **kwargs):
        fc_1 = self.fc_1(inputs)
        bbn = self.fc_2_1(fc_1)
        cbn = self.fc_2_2(fc_1)
        return bbn, cbn


class Decoder(tf.keras.layers.Layer):
    def __init__(self, middle_dim, feat_dim):
        """
        :param middle_dim: hidden units
        :param feat_dim: data dim
        """
        super(Decoder, self).__init__()
        self.fc_1 = tf.keras.layers.Dense(middle_dim, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(feat_dim)

    def call(self, inputs, **kwargs):
        fc_1 = self.fc_1(inputs)
        return self.fc_2(fc_1)
