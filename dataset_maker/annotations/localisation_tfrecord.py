from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
from annotations import dataset_utils
import tensorflow as tf

from PIL import Image
from collections import namedtuple
import contextlib2


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, class_map=None):
    with tf.gfile.GFile(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = group.filename.split(".")[-1].encode("utf8")

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    if class_map is None:
        class_map = {cls: idx for idx, cls in enumerate({row["class"] for _, row in group.object.iterrows()}, 1)}

    for index, row in group.object.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)
        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_map[row["class"]])

    return tf.train.Example(features=tf.train.Features(feature={
        "image/height": dataset_utils.int64_feature(height),
        "image/width": dataset_utils.int64_feature(width),
        "image/filename": dataset_utils.bytes_feature(filename),
        "image/source_id": dataset_utils.bytes_feature(filename),
        "image/encoded": dataset_utils.bytes_feature(encoded_jpg),
        "image/format": dataset_utils.bytes_feature(image_format),
        "image/object/bbox/xmin": dataset_utils.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_utils.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_utils.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_utils.float_list_feature(ymaxs),
        "image/object/class/text": dataset_utils.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_utils.int64_list_feature(classes)
    }))


def tensorflow_object_csv_to_tfrecord(output_path, infile, num_shards):
    with contextlib2.ExitStack() as close_stack:
        output_tfrecords = dataset_utils.open_sharded_output_tfrecords(close_stack, output_path, num_shards)
        path = os.path.join(os.getcwd())
        examples = pd.read_csv(infile)
        grouped = split(examples, "filename")
        for idx, group in enumerate(grouped):
            tf_example = create_tf_example(group, path)
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
