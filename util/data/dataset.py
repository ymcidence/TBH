import tensorflow as tf
import os
from meta import REPO_PATH
from util.data.set_processor import SET_DIM, SET_LABEL, SET_SPLIT


class ParsedRecord(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'cifar10')
        self.part_name = kwargs.get('part_name', 'train')
        self.batch_size = kwargs.get('batch_size', 256)

        self.data = self._load_data()

        self._output_types = self.data.output_types
        self._output_shapes = self.data.output_shapes

    def _load_data(self):
        def data_parser(tf_example: tf.train.Example):
            feat_dict = {'fid': tf.io.FixedLenFeature([], tf.int64),
                         'feat': tf.io.FixedLenFeature([SET_DIM.get(self.set_name, 4096)], tf.float32),
                         'label': tf.io.FixedLenFeature([SET_LABEL.get(self.set_name, 10)], tf.float32)}
            features = tf.io.parse_single_example(tf_example, features=feat_dict)

            _id = tf.cast(features['fid'], tf.int32)
            _feat = tf.cast(features['feat'], tf.float32)
            _label = tf.cast(features['label'], tf.int32)
            return _id, _feat, _label

        record_name = os.path.join(REPO_PATH + 'data', self.set_name, self.part_name + '.tfrecords')
        data = tf.data.TFRecordDataset(record_name).map(data_parser, num_parallel_calls=50).prefetch(self.batch_size)
        data = data.cache().repeat().shuffle(10000).batch(self.batch_size)

        # data = data.cache().repeat().batch(self.batch_size)

        return data

    @property
    def output_types(self):
        return self._output_types

    @property
    def output_shapes(self):
        return self._output_shapes

    @property
    def output_contents(self):
        return ['fid', 'feature', 'label']


class Dataset(object):
    def __init__(self, **kwargs):
        self.set_name = kwargs.get('set_name', 'cifar10')
        self.batch_size = kwargs.get('batch_size', 256)
        self._load_data()

    def _load_data(self):
        # 1. training data
        settings = {'set_name': self.set_name,
                    'batch_size': self.batch_size,
                    'part_name': SET_SPLIT[0]}
        self.train_data = ParsedRecord(**settings)

        # 2. test data
        settings['part_name'] = SET_SPLIT[1]
        self.test_data = ParsedRecord(**settings)
