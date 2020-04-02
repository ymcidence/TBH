from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer import binary_activation
from util.data.set_processor import SET_DIM, SET_LABEL


class JMLH(tf.keras.Model):
    def __init__(self, set_name, bbn_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_name = set_name
        self.bbn_dim = bbn_dim
        self.feat_dim = SET_DIM.get(set_name, 4096)
        self.fc_1 = tf.keras.layers.Dense(bbn_dim)
        self.fc_2 = tf.keras.layers.Dense(SET_LABEL.get(set_name, 10))

    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs[1])[0]
        fc_1 = self.fc_1(inputs[1])
        eps = tf.ones([batch_size, self.bbn_dim]) / 2.
        code, _ = binary_activation.binary_activation(fc_1, eps)
        cls = self.fc_2(code)
        return code, tf.nn.sigmoid(fc_1), cls

    def run(self, feat_in):
        batch_size = tf.shape(feat_in)[0]
        fc_1 = self.fc_1(feat_in)
        eps = tf.ones([batch_size, self.bbn_dim]) / 2.
        code, _ = binary_activation.binary_activation(fc_1, eps)
        return code
