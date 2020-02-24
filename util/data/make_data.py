import os
import tensorflow as tf
from util.data.set_processor import SET_PROCESSOR, SET_SPLIT

from meta import REPO_PATH


# noinspection PyUnusedLocal
def default_processor(root_folder):
    raise NotImplementedError


def process_mat(set_name, root_folder):
    processor = SET_PROCESSOR.get(set_name)
    return processor(root_folder)


def _int64_feature(value):
    """Create a feature that is serialized as an int64."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_tfrecord(data, set_name, part_name):
    data_length = data['feat'].shape[0]

    save_path = os.path.join(REPO_PATH, 'data', set_name)
    print(REPO_PATH)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_name = os.path.join(save_path, part_name + '.tfrecords')
    writer = tf.io.TFRecordWriter(file_name)

    for i in range(data_length):
        print(i)
        this_id = _int64_feature(data['fid'][i])
        this_feat = _float_feature(data['feat'][i, :])
        this_label = _float_feature(data['label'][i, :])
        feat_dict = {'id': this_id,
                     'feat': this_feat,
                     'label': this_label}
        feature = tf.train.Features(feature=feat_dict)
        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

    writer.close()


def build_dataset(set_name, root_folder):
    train_dict, test_dict = process_mat(set_name, root_folder)

    convert_tfrecord(train_dict, set_name, SET_SPLIT[0])
    convert_tfrecord(test_dict, set_name, SET_SPLIT[1])


if __name__ == '__main__':
    build_dataset('cifar10', '/home/ymcidence/Workspace/CodeGeass/GraphBinary/data/')
    print('hehe')
