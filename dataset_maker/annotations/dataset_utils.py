import tensorflow as tf
import numpy as np
import contextlib2
from typing import List
from download_upload import LoaderDownloader


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


def open_sharded_tfrecords(exit_stack: contextlib2.ExitStack, base_path: str, num_shards: int) \
        -> List[contextlib2.ExitStack]:
    """
    Open exits stacks to enable the writing of files over several shards.
    :param exit_stack: THe stack to communicate where the shard is being written.
    :param base_path: The base path for the shard path.
    :param num_shards: The number of shards being created.
    :return: List of exit stacks to store where each the shards are being written.
    """
    return [
        exit_stack.enter_context(tf.io.TFRecordWriter(f"{base_path}-{idx:05d}-of-{num_shards:05d}"))
        for idx in range(1, num_shards + 1)
    ]


def open_sharded_tfrecords_with_splits(exit_stack: contextlib2.ExitStack, base_path: str, num_shards: int,
                                       shard_splits: List[float], split_names: List[str]) \
        -> List[contextlib2.ExitStack]:
    """
    Open exits stacks to enable the writing of files over several shards.
    :param exit_stack: THe stack to communicate where the shard is being written.
    :param base_path: The base path for the shard path.
    :param num_shards: The number of shards being created.
    :param shard_splits: The ratios at which the shards are split.
    :param split_names: The names of each the shard splits.
    :return: List of exit stacks to store where each the shards are being written.
    :raise AssertionError: If the sum of the ratios is less than 1.
    :raise AssertionError: If there are less names then the number of splits.
    """
    assert sum(shard_splits) <= 1.0, \
        f"The sum of shard_split need to be less than or equal 1. (! {sum(shard_splits)} <= 1.0)"
    if split_names is not None:
        assert len(split_names) >= len(shard_splits), \
            "There need to be more shard names then there shards to be named if they are to be named."
    else:
        split_names = [f"split_{i}" for i in range(len(shard_splits))]

    splits = [int((sum(shard_splits[:i]) + a) * num_shards) for i, a in enumerate(shard_splits)]
    return [
        exit_stack.enter_context(
            tf.python_io.TFRecordWriter(f"{base_path}/{split_names[sidx]}-{idx:05d}-of-{num_shards:05d}")
        )
        for sidx, s in enumerate(np.split(np.arange(1, num_shards), splits)) for idx in s
    ]


def annotation_format_converter(registry):
    def wrapper(image_dir, annotations_dir, download_dir, in_format, out_format) -> None:
        """
        Converts annotation from one format to another.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: The directory of the annotations file.
        :param download_dir: The directory where the annotations are being downloaded.
        :param in_format: The name of the format being converted from.
        :param out_format: The name of the format being converted to.
        :raise AssertionError: If in_format is not string or InstanceSegmentationAnnotation.
        :raise AssertionError: If out_format is not string or InstanceSegmentationAnnotation.
        """
        assert issubclass(registry, LoaderDownloader), 'The registry must be an instance of LoaderDownloader.'

        assert isinstance(in_format, (registry, str)), \
            f'in_format: {in_format} need to string or {type(registry)}.'
        assert isinstance(out_format, (registry, str)), \
            f'out_format: {out_format} need to string or {type(registry)}.'

        if isinstance(in_format, str):
            in_format = registry(in_format)

        if isinstance(out_format, str):
            out_format = registry(out_format)

        out_format.download(download_dir, *in_format.load(image_dir, annotations_dir))
    return wrapper

