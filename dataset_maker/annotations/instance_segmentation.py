from dataset_maker.annotations.download_upload import ShardedConvert
from dataset_maker.patterns import SingletonStrategies, strategy_method
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Generator, Sized
import numpy as np
import matplotlib.pyplot as plt
from dataset_maker import utils
import json
import os
import io
import tensorflow as tf
from dataset_maker.annotations import dataset_utils, vgg_utils
import contextlib2
from PIL import Image
from collections import defaultdict
from itertools import count


class InstanceSegmentationAnnotationFormats(SingletonStrategies):
    """
   Singleton for holding instance segmentation annotation formats.
   """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Annotations formats: \n" + \
               "\n".join([f"{i:3}: {n}" for i, (n, _) in enumerate(self.strategies.values())])


FORMATS = InstanceSegmentationAnnotationFormats()


class InstanceSegmentationAnnotation(ShardedConvert, metaclass=ABCMeta):
    """
    Abstract base class for InstanceSegmentationAnnotation as a Loader.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def load(image_dir: str, annotations_file: str) -> \
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        return super().load(image_dir, annotations_file)

    @staticmethod
    def download(download_dir, image_names, images, bboxes, polygons, classes) -> None:
        pass

    def create_tfrecord(self, image_dir: str, annotations_file: str, output_dir, num_shards=1, class_map=None):
        filenames, images, bboxes, polygons, classes = self.load(image_dir, annotations_file)
        if class_map is None:
            unique_classes = {cls for cls_per in classes for cls in cls_per}
            class_map = {cls: idx for idx, cls in enumerate(unique_classes, 1)}

        with contextlib2.ExitStack() as close_stack:
            output_tfrecords = dataset_utils.open_sharded_tfrecords(close_stack, output_dir, num_shards)

            for idx, (filename, image, bbox_per, poly_per, cls_per) in \
                    enumerate(zip(filenames, images, bboxes, polygons, classes)):
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

                for (y0, x0, y1, x1), poly, cls in zip(bbox_per, poly_per, cls_per):
                    ymins.append(float(y0 / height))
                    xmins.append(float(x0 / width))
                    ymaxs.append(float(y1 / height))
                    xmaxs.append(float(x1 / width))

                    mask = utils.polygon_to_mask(*poly, width, height)
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
                    "image/object/mask": dataset_utils.bytes_list_feature(encode_masks)
                }))

                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())


@strategy_method(InstanceSegmentationAnnotationFormats)
class VGG(InstanceSegmentationAnnotation):
    """
    Instance Segmentation Annotation Class for the loading and downloading VGG annotations. VGG Annotation for use a .json
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
    def _load_shards(num_shards, image_dir: str, annotations_dir: str, region_label: str = "label") -> Generator:
        def _load_shard(shard):
            names = []
            images = []
            bboxes = []
            polygons = []
            classes = []
            for annotation in shard.values():
                filename = annotation["filename"]
                names.append(filename)

                image = plt.imread(f"{image_dir}/{filename}")
                images.append(image)

                bboxes_per = []
                poly_per = []
                classes_per = []

                regions = annotation["regions"]
                if isinstance(regions, dict):
                    regions = regions.values()

                for r in regions:
                    xs, ys = r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"]
                    bbox = utils.bbox(xs, ys)
                    bboxes_per.append(np.asarray(bbox))
                    poly_per.append((xs, ys))
                    classes_per.append(r["region_attributes"][region_label])
                bboxes.append(np.asarray(bboxes_per))
                polygons.append(np.asarray(poly_per))
                classes.append(np.asarray(classes_per))
            return names, images, bboxes, polygons, classes

        if annotations_dir.endswith(".json"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
            assert len(potential_annotations) != 0, \
                f"There is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"There are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]
            annotations_file = f"{annotations_dir}/{annotations_file}"

        with open(annotations_file, "r") as f:
            annotations = json.load(f)
            annotations = vgg_utils.convert_annotations_to_polygon(annotations)

        return (_load_shard(c) for c in utils.chunks(annotations, round(len(annotations) / num_shards)))


    @staticmethod
    def _process_shard(image_names, images, bboxes, polygons, classes) -> Sized:
        assert len(image_names) == len(images) == len(bboxes) == len(polygons) == len(classes), \
            "The params image_names, images, bboxes, polygons and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(polygons): {len(polygons)}" \
            f"len(classes): {len(classes)}"
        return {
            name: {
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [int(x) for x in xs],
                            "all_points_y": [int(y) for y in ys]},
                        "region_attributes": {"label": str(cls)}
                    }
                    for cls, (xs, ys) in zip(classes_per, poly_per)
                ]
            }
            for name, image, poly_per, classes_per in zip(image_names, images, polygons, classes)
        }



    @staticmethod
    def _combine_shard_output(shards):
        comb = {}
        for s in shards:
            comb.update(s)
        return s

    @staticmethod
    def _write(download_dir: str, data: dict) -> None:
        with open(f"{download_dir}/vgg_annotations.json", "w") as f:
            json.dump(data, f)


@strategy_method(InstanceSegmentationAnnotationFormats)
class COCO(InstanceSegmentationAnnotation):
    """
    Instance Segmentation Annotation Class for the loading and downloading COCO annotations. COCO Annotation for use a .json
    format. For Example:
    {
        "info": {
            "images": [
                {
                    "id": 1,
                    "width": 1504,
                    "height": 2016,
                    "file_name": "image_1.jpg"
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "iscrowd": 0,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation":[[87.281, 708.408, 1416.71826, 1307.59]],
                    "bbox": [87.281, 708.408, 1416.71826, 1307.59],
                    "area":796574.87632
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "laptop"
                }
            ]
        }
    """

    @staticmethod
    def _load_shards(num_shards, image_dir, annotations_dir) -> Generator:
        def _load_shard(shard):
            image_dict = defaultdict(lambda: {"bboxes": [], "classes": [], "polygons": []})

            for name, annotations_per in shard.items():
                image = plt.imread(f"{image_dir}/{name}")
                h, w, *_ = image.shape

                image_dict[name]["file_name"] = name
                image_dict[name]["image"] = image

                for a in annotations_per:
                    x0, y0, bb_width, bb_height = a["bbox"]
                    x1, y1 = x0 + bb_width, y0 + bb_height
                    image_dict[name]["bboxes"].append(np.asarray([y0, x0, y1, x1], dtype="int64"))
                    image_dict[name]["classes"].append(classes_dict[a["category_id"]])

                    # TODO Implement workflow to allow pycocotools to be installed
                    if a["iscrowd"]:
                        raise NotImplementedError()
                    else:
                        segmentation = a["segmentation"][0]
                        poly = segmentation[::2], segmentation[1::2]
                    image_dict[name]["polygons"].append(poly)

            names = []
            images = []
            bboxes = []
            polygons = []
            classes = []
            for info in image_dict.values():
                name = info["file_name"]
                names.append(name)
                images.append(info["image"])
                bboxes.append(np.asarray(info["bboxes"]))
                polygons.append(info["polygons"])
                classes.append(np.asarray(info["classes"]))
            return names, images, bboxes, polygons, classes

        if annotations_dir.endswith(".json"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
            assert len(potential_annotations) != 0, \
                f"There is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"There are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]
            annotations_file = f"{annotations_dir}/{annotations_file}"

        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        classes_dict = {cls_info["id"]: cls_info["name"] for cls_info in annotations["categories"]}
        image_idx_name = {image_info["id"]: image_info["file_name"] for image_info in annotations["images"]}

        annotations_per_image = defaultdict(list)
        for a in annotations["annotations"]:
            annotations_per_image[image_idx_name[a["image_id"]]].append(a)
        return (_load_shard(c) for c in
                utils.chunks(annotations_per_image, round(len(annotations_per_image) / num_shards)))


    @staticmethod
    def _process_shard(image_names, images, bboxes, polygons, classes) -> Sized:
        assert len(image_names) == len(images) == len(bboxes) == len(polygons) == len(classes), \
            "The params image_names, images, bboxes, polygons and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(polygons): {len(polygons)}" \
            f"len(classes): {len(classes)}"
        classes_dict = {n: i for i, n in enumerate({cls for classes_per in classes for cls in classes_per}, 1)}

        images_info = {}
        annotations_info = []
        classes_info = set()
        anno_idx = count()
        for img_idx, (name, image, bboxes_per, poly_per, classes_per) in \
                enumerate(zip(image_names, images, bboxes, polygons, classes), 1):
            h, w, _ = image.shape

            images_info[str(name)] = {"id": img_idx, "file_name": str(name), "width": int(w), "height": int(h)}

            for (y0, x0, y1, x1), (xs, ys), cls in zip(bboxes_per, poly_per, classes_per):
                annotations_info.append({
                    "id": next(anno_idx),
                    "image_id": str(name),
                    "category_id": str(classes_dict[cls]),
                    "iscrowd": 0,
                    "segmentation": [[int(p) for ps in zip(xs, ys) for p in ps]],
                    "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                    "area": int(utils.bbox_area(y0, x0, y1, x1))
                })
                classes_info |= set(classes_per)
        return {
            "images": images_info,
            "annotations": annotations_info,
            "categories": [{"id": int(cat_idx), "name": str(cls), "supercategory": "type"}
                           for cls, cat_idx in classes_dict.items()]
        }

    @staticmethod
    def _combine_shard_output(shards):
        cls_idx, img_idx, anno_idx = count(), count(), count()

        # TODO work out if slow
        categories = {cls["name"]: {"id": next(cls_idx), "name": str(cls), "supercategory": "type"}
                      for s in shards for cls in s["categories"]}
        image_info = {info["file_name"]: {"id": next(img_idx), **info}
                      for s in shards for info in s["images"].values()}

        annotations_info = []
        for s in shards:
            local_cls_idx = {cls["id"]: categories[cls["name"]]["id"] for cls in s["categories"]}
            local_img_idx = {info["id"]: image_info[info["file_name"]]["id"] for info in s["images"]}
            for a in s["annotations"]:
                a["category_id"] = local_cls_idx[a["category_id"]]
                a["image_id"] = local_img_idx[a["image_id"]]
                a["id"] = next(anno_idx)
                annotations_info.append(a)

        return {
            "images": list(image_info.values()),
            "annotations": annotations_info,
            "categories": list(categories.values())
        }

    @staticmethod
    def _write(download_dir: str, data: dict) -> None:
        with open(f"{download_dir}/coco_annotations.json", "w") as f:
            json.dump(data, f)


convert_annotation_format = dataset_utils.annotation_format_converter(InstanceSegmentationAnnotation, FORMATS)
