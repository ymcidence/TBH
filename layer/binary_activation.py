from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


def sigmoid_sign(logits, eps):
    """
    {0,1} sign function with (1) sigmoid activation (2) perturbation of eps in sigmoid
    :param logits: bottom layer output
    :param eps: randomly sampled values between [0,1]
    :return:
    """
    prob = 1.0 / (1 + tf.exp(-logits))
    code = (tf.sign(prob - eps) + 1.0) / 2.0
    return code, prob


@tf.custom_gradient
def binary_activation(logits, eps):
    """

    :param logits:
    :param eps:
    :return:
    """
    code, prob = sigmoid_sign(logits, eps)

    def grad(_d_code, _d_prob):
        """
        Distributional derivative with Bernoulli probs
        :param _d_code: bp gradients through code
        :param _d_prob: bp gradients through prob
        :return:
        """

        d_logits = prob * (1 - prob) * (_d_code + _d_prob)

        d_eps = _d_code
        return d_logits, d_eps

    return [code, prob], grad


if __name__ == '__main__':
    a = tf.constant([0.1, 0.2, -1, -0.7], dtype=tf.float32)
    b = tf.random.uniform([4])
    with tf.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        c, cc = binary_activation(a, b)
        d = tf.reduce_sum(c)

    e = tape.gradient(target=c, sources=a)
    print(b)
    print(c, e)
    print(b)
    print(b)
