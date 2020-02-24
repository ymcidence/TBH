import tensorflow as tf


class ArrayReader(object):
    def __init__(self, set_name='1', batch_size=256, pre_process=False):
        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.compat.v1.Session(config=config)
        self.set_name = set_name
        self.batch_size = batch_size
        self.pre_process = pre_process
        self.data = self._build_data()

    def _build_data(self):
        raise NotImplementedError()

    def get_batch(self, part='training'):
        assert hasattr(self.data, part + '_handle')
        assert hasattr(self.data, 'train_test_handle')
        assert hasattr(self.data, 'feed')

        handle = getattr(self.data, part + '_handle')

        batch_data = self.sess.run(self.data.feed, feed_dict={self.data.train_test_handle: handle})
        return batch_data

    def get_batch_tensor(self, part='training'):
        pass
