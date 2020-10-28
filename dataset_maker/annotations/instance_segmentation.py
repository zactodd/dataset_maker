import csv
import hashlib
from dataset_maker.patterns import SingletonStrategies, strategy_method
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Union
import numpy as np
from xml.etree import ElementTree
import matplotlib.pyplot as plt
from functools import reduce
from dataset_maker import utils
import json
import re
import os
import io
from collections import defaultdict
import tensorflow as tf
from dataset_maker.annotations import dataset_utils
import contextlib2
from PIL import Image


class InstanceSegmentationAnnotationFormats(SingletonStrategies):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Annotations formats: \n" + "\n".join([f"{i:3}: {k}" for i, k in enumerate(self.strategies.keys())])


class InstanceSegmentationAnnotation(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def load(image_dir: str, annotations_file: str) ->\
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        pass

    @staticmethod
    @abstractmethod
    def download(download_dir, image_names, images, bboxes, masks, classes) -> None:
        pass

    def create_tfrecord(self, image_dir: str, annotations_file: str, output_dir, num_shards=1, class_map=None):
        filenames, images, bboxes, masks, classes = self.load(image_dir, annotations_file)
        if class_map is None:
            unique_classes = {cls for cls_per in classes for cls in cls_per}
            class_map = {cls: idx for idx, cls in enumerate(unique_classes, 1)}

        with contextlib2.ExitStack() as close_stack:
            output_tfrecords = dataset_utils.open_sharded_output_tfrecords(close_stack, output_dir, num_shards)

            for idx, (filename, image, bbox_per, mask_per, cls_per) in \
                    enumerate(zip(filenames, images, bboxes, masks, classes)):
                # TODO maybe look into different way or find the common standard

                with tf.io.gfile.GFile(f"{image_dir}/{filename}", "rb") as fid:
                    encoded_image = fid.read()

                image = Image.fromarray(np.uint8(image * 255))
                width, height = image.size

                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                encode_masks = []
                classes_text = []
                mapped_classes = []

                for (y0, x0, y1, x1), mask, cls in zip(bbox_per, mask_per, cls_per):
                    ymins.append(float(y0 / height))
                    xmins.append(float(x0 / width))
                    ymaxs.append(float(y1 / height))
                    xmaxs.append(float(x1 / width))

                    mask_image = Image.fromarray(mask)
                    output = io.BytesIO()
                    mask_image.save(output, format='PNG')
                    encode_masks.append(output.getvalue())

                    classes_text.append(cls.encode("utf8"))
                    mapped_classes.append(class_map[cls])

                image_format = filename.split(".")[-1].encode("utf8")
                encode_filename = filename.encode("utf8")

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    "image/height": dataset_utils.int64_feature(height),
                    "image/width": dataset_utils.int64_feature(width),
                    "image/filename": dataset_utils.bytes_feature(encode_filename),
                    "image/source_id": dataset_utils.bytes_feature(encode_filename),
                    "image/encoded": dataset_utils.bytes_feature(encoded_image),
                    "image/format": dataset_utils.bytes_feature(image_format),
                    "image/object/bbox/xmin": dataset_utils.float_list_feature(xmins),
                    "image/object/bbox/xmax": dataset_utils.float_list_feature(xmaxs),
                    "image/object/bbox/ymin": dataset_utils.float_list_feature(ymins),
                    "image/object/bbox/ymax": dataset_utils.float_list_feature(ymaxs),
                    "image/object/class/text": dataset_utils.bytes_list_feature(classes_text),
                    "image/object/class/label": dataset_utils.int64_list_feature(mapped_classes),
                    "image/object/mask": dataset_utils.bytes_list_feature(masks)
                }))

                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
