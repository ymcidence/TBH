from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layer import gcn


@tf.function
def build_adjacency_hamming(tensor_in):
    """
    Hamming-distance-based graph. It is self-connected.
    :param tensor_in: [N D]
    :return:
    """
    code_length = tf.cast(tf.shape(tensor_in)[1], tf.float32)
    m1 = tensor_in - 1
    c1 = tf.matmul(tensor_in, m1, transpose_b=True)
    c2 = tf.matmul(m1, tensor_in, transpose_b=True)
    normalized_dist = tf.math.abs(c1 + c2) / code_length
    return tf.pow(1 - normalized_dist, 1.4)


# noinspection PyAbstractClass
class TwinBottleneck(tf.keras.layers.Layer):
    def __init__(self, bbn_dim, cbn_dim, **kwargs):
        super().__init__(**kwargs)
        self.bbn_dim = bbn_dim
        self.cbn_dim = cbn_dim
        self.gcn = gcn.GCNLayer(cbn_dim)

    # noinspection PyMethodOverriding
    def call(self, bbn, cbn):
        adj = build_adjacency_hamming(bbn)
        return tf.nn.sigmoid(self.gcn(cbn, adj))
