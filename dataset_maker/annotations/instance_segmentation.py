from dataset_maker.annotations.download_upload import Loader
from dataset_maker.patterns import SingletonStrategies, strategy_method
from abc import ABCMeta, abstractmethod
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from dataset_maker import utils
import json
import os
import io
import tensorflow as tf
from dataset_maker.annotations import dataset_utils
import contextlib2
from PIL import Image


class InstanceSegmentationAnnotationFormats(SingletonStrategies):
    """
   Singleton for holding instance segmentation annotation formats.
   """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Annotations formats: \n" + \
               "\n".join([f"{i:3}: {n}" for i, (n, _) in enumerate(self.strategies.values())])


class InstanceSegmentationAnnotation(Loader, metaclass=ABCMeta):
    """
    Abstract base class for InstanceSegmentationAnnotation as a Loader.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def load(image_dir: str, annotations_file: str) -> \
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


@strategy_method(InstanceSegmentationAnnotationFormats)
class VGG(InstanceSegmentationAnnotation):
    """
    Instance Segmenation Annotation Class for the loading and downloading VGG annotations. VGG Annotation for use a .json
    format. VGG can have its 'regions' as a dictionary with a full VGG file looking like:
    {
        "image_1.png": {
                "regions": {
                        0: {
                            "shape_attributes": {
                                "name": "polygon",
                                "all_points_x": [0, 25, 25, 0],
                                "all_points_y": [0, 0, 25, 25]
                            },
                            "region_attributes": {"label": "catfish"}
                        }
        }
    }
    And VGG can have its 'regions' as a list with a full VGG file looking like:
    {
        "image_1.png": {
                "regions": [
                                {
                                    "shape_attributes": {
                                        "name": "polygon",
                                        "all_points_x": [0, 25, 25, 0],
                                        "all_points_y": [0, 0, 25, 25]
                                    },
                                    "region_attributes": {"label": "catfish"}
                                }
                ]
        }
    }
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str, region_label: str = "label") -> \
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Loads a VGG file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :param region_label: The key that identifies the label being loaded.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith(".json"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
            assert len(potential_annotations) != 0, \
                f"There is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"There are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]
        with open(f"{annotations_dir}/{annotations_file}", "r") as f:
            annotations = json.load(f)

        names = []
        images = []
        bboxes = []
        masks = []
        classes = []
        for filename, annotation in annotations.items():
            names.append(filename)

            image = plt.imread(f"{image_dir}/{filename}")
            w, h, _ = image.shape
            images.append(image)

            bboxes_per = []
            masks_per = []
            classes_per = []

            regions = annotation["regions"]
            if isinstance(regions, dict):
                regions = regions.values()

            for r in regions:
                xs, ys = r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]
                bbox = utils.bbox(xs, ys)
                bboxes_per.append(np.asarray(bbox))
                masks_per.append(utils.polygon_to_mask(xs, ys, w, h))
                classes_per.append(r["region_attributes"][region_label])
            bboxes.append(np.asarray(bboxes_per))
            masks.append(np.asarray(masks_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, masks, classes


@strategy_method(InstanceSegmentationAnnotationFormats)
class COCO(InstanceSegmentationAnnotation):
    @staticmethod
    def load(image_dir: str, annotations_dir: str) ->\
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Loads a COCO file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith(".json"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
            assert len(potential_annotations) != 0, \
                f"There is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"There are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]
        with open(f"{annotations_dir}/{annotations_file}", "r") as f:
            annotations = json.load(f)

        classes_dict = {cls_info["id"]: cls_info["name"] for cls_info in annotations["categories"]}

        image_dict = {
            image_info["id"]: {
                "bboxes": [],
                "classes": [],
                "masks": [],
                "name": image_info["filename"],
                "image": plt.imread(f"{image_dir}/{image_info['filename']}")
            }
            for image_info in annotations["images"]
        }
        for annotation in annotations["annotations"]:
            idx = annotation["image_id"]
            x0, y0, x1, y1 = annotation["bbox"]
            image_dict[idx]["bboxes"].append(np.asarray([y0, x0, y1, x1], dtype="int64"))
            image_dict[idx]["classes"].append(classes_dict[annotation["category"]])

            h, w, *_ = image_dict[idx]["image"].shape

            # TODO Implement workflow to allow pycocotools to be installed
            if annotation["iscrowd"]:
                raise NotImplementedError()
                # mask = utils.rle_to_mask(annotation["segmentation"])
            else:
                segmentation = annotation["segmentation"][0]
                x, y = segmentation[::2], segmentation[1::2]
                mask = utils.polygon_to_mask(x, y, w, h)
            image_dict[idx]["masks"].append(mask)

        names = []
        images = []
        bboxes = []
        masks = []
        classes = []
        for info in image_dict.values():
            name = info["name"]
            names.append(name)
            images.append(info["image"])
            bboxes.append(np.asarray(info["bboxes"]))
            masks.append(info["masks"])
            classes.append(np.asarray(info["classes"]))
        return names, images, masks, bboxes, classes
