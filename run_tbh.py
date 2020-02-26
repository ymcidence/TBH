from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from train import tbh_train

tbh_train.train('cifar10', 32, 512, 256)