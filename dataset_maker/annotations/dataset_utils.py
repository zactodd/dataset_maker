import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    return [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(f"{base_path}-{idx:05d}-of-{num_shards:05d}"))
        for idx in range(1, num_shards + 1)
    ]
