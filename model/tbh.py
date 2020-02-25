from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer import encodec, twin_bottleneck
from util.data.set_processor import SET_DIM


# noinspection PyAbstractClass
class TBH(tf.keras.Model):
    def __init__(self, set_name, bbn_dim, cbn_dim, middle_dim=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_name = set_name
        self.bbn_dim = bbn_dim
        self.cbn_dim = cbn_dim
        self.middle_dim = middle_dim
        self.feat_dim = SET_DIM.get(set_name, 4096)

        self.encoder = encodec.Encoder(middle_dim, bbn_dim, cbn_dim)
        self.decoder = encodec.Decoder(middle_dim, self.feat_dim)
        self.tbn = twin_bottleneck.TwinBottleneck(bbn_dim, cbn_dim)

        self.dis_1 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.dis_2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        feat_in = tf.cast(inputs[0][1], dtype=tf.float32)
        bbn, cbn = self.encoder(feat_in, training=training)

        if training:
            bn = self.tbn(bbn, cbn)
            dis_1 = self.dis_1(bbn)
            dis_2 = self.dis_2(bn)
            feat_out = self.decoder(bn)
            sample_bbn = inputs[1]
            sample_bn = inputs[2]
            dis_1_sample = self.dis_1(sample_bbn)
            dis_2_sample = self.dis_2(sample_bn)
            return bbn, feat_out, dis_1, dis_2, dis_1_sample, dis_2_sample
        else:
            return bbn
