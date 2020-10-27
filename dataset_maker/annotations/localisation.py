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
from collections import defaultdict
import tensorflow as tf
from dataset_maker.annotations import dataset_utils
import contextlib2
from PIL import Image
import io


IMAGE_FORMATS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")


class LocalisationAnnotationFormats(SingletonStrategies):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Annotations formats: \n" + "\n".join([f"{i:3}: {k}" for i, k in enumerate(self.strategies.keys())])


class LocalisationAnnotation(ABC):
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def load(image_dir: str, annotations_file: str) -> Dict:
        pass

    @staticmethod
    @abstractmethod
    def download(download_dir, image_names, images, bboxes, classes) -> None:
        pass

    @staticmethod
    @abstractmethod
    def create_tfrecord(image_dir: str, annotations_file: str, output_dir) -> Dict:
        pass


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class VGG:
    """
    Localisation Annotation Class for the loading and downloading VGG annotations. VGG Annotation for use a .json
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
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Loads a VGG file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the 
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise ValueError: If there is more than one json file in the directory of :param annotations_dir.
        :raise ValueError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith(".json"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
            assert len(potential_annotations) != 0, \
                f"Theres is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"Theres are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]
        with open(f"{annotations_dir}/{annotations_file}", "r") as f:
            annotations = json.load(f)

        names = []
        images = []
        bboxes = []
        classes = []
        for filename, annotation in annotations.items():
            names.append(filename)
            images.append(plt.imread(f"{image_dir}/{filename}"))

            bboxes_per = []
            classes_per = []
            
            regions = annotation["regions"]
            if isinstance(regions, dict):
                regions = regions.values()
                
            for r in regions:
                bbox = utils.bbox(r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"])
                bboxes_per.append(np.asarray(bbox))
                classes_per.append(r["region_attributes"]["label"])
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes:  List[np.ndarray]) -> None:
        """
        Downloads a VGG json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        annotations = {
            name: {
                "filename": name,
                "regions": {
                    str(i): {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [int(x0), int(x1), int(x1), int(x0)],
                            "all_points_y": [int(y0), int(y0), int(y1), int(y1)]
                        },
                        "region_attributes": {"label": str(cls)}
                    }
                    for i, ((y0, x0, y1, x1), cls) in enumerate(zip(bboxes_per, classes_per))
                }
            }
            for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes)
        }
        with open(f"{download_dir}/vgg_annotations.json", "w") as f:
            json.dump(annotations, f)

    @staticmethod
    def create_tfrecord(image_dir: str, annotations_file: str, output_dir) -> Dict:
        pass


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class PascalVOC:
    """
    Localisation Annotation Class for the loading and downloading Pascal VOC annotations.
    Pascal VOC annotations are xml per image being annotated. For example:
    <annotation>
        <filename>image_1.jpg</filename>
        <size>
            <width>2048</width>
            <height>1536</height>
            <depth>3</depth>
        </size>
        <object>
            <name>dog</name>
            <pose>Unspecified</pose>
            <truncated>Unspecified</truncated>
            <difficult>Unspecified</difficult>
            <bndbox>
                <xmin>293</xmin>
                <ymin>384</ymin>
                <xmax>2055</xmax>
                <ymax>1518</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    
    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> Tuple[list, list, list, list]:
        """
        Loads a Pascal VOC xml files and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: The directory of the annotations file.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        """
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
        names = []
        images = []
        bboxes = []
        classes = []
        for f in annotation_files:
            root = ElementTree.parse(f"{annotations_dir}/{f}")
            name = root.find("filename").text
            names.append(name)
            images.append(plt.imread(f"{image_dir}/{name}"))
            bboxes_per = []
            classes_per = []
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                y0 = int(bbox.find("ymin").text)
                x0 = int(bbox.find("xmin").text)
                y1 = int(bbox.find("ymax").text)
                x1 = int(bbox.find("xmax").text)
                bboxes_per.append(np.asarray([y0, x0, y1, x1]))
                classes_per.append(obj.find("name").text)
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes:  List[np.ndarray]) -> None:
        """
        Downloads a Pascal VOC xml files to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        folder = re.split("/|\\\\", download_dir)[-1]
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            w, h, d = image.shape

            root = ElementTree.Element("annotation")
            ElementTree.SubElement(root, "folder").text = folder
            ElementTree.SubElement(root, "filename").text = name

            size = ElementTree.SubElement(root, "size")
            ElementTree.SubElement(size, "width").text = str(w)
            ElementTree.SubElement(size, "height").text = str(h)
            ElementTree.SubElement(size, "depth").text = str(d)

            for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                obj = ElementTree.SubElement(root, "object")
                ElementTree.SubElement(obj, "name").text = str(cls)

                ElementTree.SubElement(obj, "pose").text = "Unspecified"
                ElementTree.SubElement(obj, "truncated").text = "Unspecified"
                ElementTree.SubElement(obj, "difficult").text = "Unspecified"

                bb_elm = ElementTree.SubElement(obj, "bndbox")
                ElementTree.SubElement(bb_elm, "xmin").text = str(x0)
                ElementTree.SubElement(bb_elm, "ymin").text = str(y0)
                ElementTree.SubElement(bb_elm, "xmax").text = str(x1)
                ElementTree.SubElement(bb_elm, "ymax").text = str(y1)

            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f"{download_dir}/{save_name}.xml", "wb") as f:
                f.write(ElementTree.tostring(root))

    @staticmethod
    def create_tfrecord(image_dir, annotations_dir, output_path, num_shards=1, class_map=None):
        def _create_example(image_dir, filename, annotation_path, class_map):
            with tf.gfile.GFile(f"{image_dir}/{filename}", "rb") as fid:
                encoded_image = fid.read()
                encoded_io = io.BytesIO(encoded_image)
                image = Image.open(encoded_io)
                width, height = image.size

            encode_filename = filename.encode("utf8")
            image_format = filename.split(".")[-1].encode("utf8")

            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []

            root = ElementTree.parse(annotation_path)
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                ymins.append(float(bbox.find("ymin").text) / height)
                xmins.append(float(bbox.find("xmin").text) / width)
                ymaxs.append(float(bbox.find("ymax").text) / height)
                xmaxs.append(float(bbox.find("xmax").text) / width)

                cls = obj.find("name").text
                classes_text.append(cls.encode("utf8"))
                classes.append(class_map[cls])

            return tf.train.Example(features=tf.train.Features(feature={
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
                "image/object/class/label": dataset_utils.int64_list_feature(classes)
            }))

        annotation_files = [f"{annotations_dir}/{f}" for f in os.listdir(annotations_dir) if f.endswith(".xml")]

        if class_map is None:
            classes = {obj.find("name").text for f in annotation_files
                       for obj in ElementTree.parse(f).findall("object")}
            class_map = {cls: idx for idx, cls in enumerate(classes, 1)}

        with contextlib2.ExitStack() as close_stack:
            output_tfrecords = dataset_utils.open_sharded_output_tfrecords(close_stack, output_path, num_shards)

            for idx, f in enumerate(annotation_files):
                root = ElementTree.parse(f)
                image_file = root.find("filename").text
                tf_example = _create_example(image_dir, image_file, f, class_map)
                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class COCO:
    """
    Localisation Annotation Class for the loading and downloading COCO annotations. COCO Annotation for use a .json
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
            ]
            "categories": [
                {
                    "id": 1,
                    "name": "laptop"
                }
            ]
        }
    """
    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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
        :raise ValueError: If there is more than one json file in the directory of :param annotations_dir.
        :raise ValueError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith(".json"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
            assert len(potential_annotations) != 0, \
                f"Theres is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"Theres are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]
        with open(f"{annotations_dir}/{annotations_file}", "r") as f:
            annotations = json.load(f)

        classes_dict = {cls_info["id"]: cls_info["name"] for cls_info in annotations["categories"]}

        image_dict = {
            image_info["id"]: {"bboxes": [], "classes": [], "name": image_info["filename"]}
            for image_info in annotations["images"]
        }
        for annotation in annotations["annotations"]:
            idx = annotation["image_id"]
            x0, y0, x1, y1 = annotation["bbox"]
            image_dict[idx]["bboxes"].append(np.asarray([y0, x0, y1, x1], dtype="int64"))
            image_dict[idx]["classes"].append(classes_dict[annotation["category"]])

        names = []
        images = []
        bboxes = []
        classes = []
        for info in image_dict.values():
            name = info["name"]
            names.append(name)
            images.append(plt.imread(f"{image_dir}/{name}"))
            bboxes.append(np.asarray(info["bboxes"]))
            classes.append(np.asarray(info["classes"]))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a COCO json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        classes_dict = {n: i for i, n in enumerate({cls for classes_per in classes for cls in classes_per}, 1)}
        annotation_idx = 0
        images_info = []
        annotations_info = []
        for img_idx, (name, image, bboxes_per, classes_per) in enumerate(zip(image_names, images, bboxes, classes), 1):
            w, h, _ = image.shape
            images_info.append({"id": img_idx, "filename": name, "width": int(w), "height": int(h)})
            for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                bbox = [float(x0), float(y0), float(x1), float(y1)]
                annotations_info.append({
                    "id": annotation_idx,
                    "image_id": img_idx,
                    "category": classes_dict[cls],
                    "iscrowd": 0,
                    "segmentation": [bbox],
                    "bbox": bbox,
                    "area": float(utils.bbox_area(y0, x0, y1, x1))
                })
                annotation_idx += 1

        data = {
            "images": images_info,
            "annotations": annotations_info,
            "categories": [{"id": cat_idx, "name": str(cls)} for cls, cat_idx in classes_dict.items()]
        }
        with open(f"{download_dir}/coco_annotations.json", "w") as f:
            json.dump(data, f)

    @staticmethod
    def create_tfrecord(image_dir: str, annotations_file: str, output_dir) -> Dict:
        pass


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class YOLO:
    """

    Localisation Annotation Class for the loading and downloading YOLO annotations.
    YOLO annotations are txt file per image being annotated. For example:
    0 0.573204 0.619149 0.860499 0.738120
    1 0.758543 0.532122 0.241968 0.665306
    """
    @staticmethod
    def load(image_dir, annotations_dir) -> Tuple[list, list, list, list]:
        """
        Loads a YOLO txt files and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: The directory of the annotations file.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise OSError: If there are no image files corresponding to an annotations txt filename.
        :raise OSError: If there more than one image file corresponding to an annotations txt filename.
        """
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]
        names = []
        images = []
        bboxes = []
        classes = []
        for file in annotation_files:
            file_path = f"{annotations_dir}/{file}"
            with open(file_path, "r") as f:
                potential_images = []
                for fmt in IMAGE_FORMATS:
                    image_path = f"{image_dir}/{file.strip('.txt')}{fmt}"
                    if os.path.exists(image_path):
                        potential_images.append(image_path)

                assert len(potential_images) != 0, \
                    f"Theres is no image file in {image_dir} corresponding to the YOLO file {file_path}."
                assert len(potential_images) == 1, \
                    f"Theres are too many image file in {image_dir} corresponding to the YOLO file {file_path}."

                image_path = potential_images[0]
                name = re.split("/|\\\\", image_path)[-1]
                names.append(name)

                image = plt.imread(image_path)
                images.append(images)
                w, h, _ = image.shape

                bboxes_per = []
                classes_per = []
                for line in f.readlines():
                    cls, *bbox = line.split()
                    x0, y0, dx, dy = [float(p) for p in bbox]
                    bboxes_per.append(np.asarray([y0 * h, x0 * w, (y0 + dy) * h, (x0 + dx) * w], dtype="int64"))
                    classes_per.append(cls)
                bboxes.append(np.asarray(bboxes_per))
                classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a YOLO txt files to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        classes_dict = {n: i for i, n in enumerate({cls for classes_per in classes for cls in classes_per})}
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f"{download_dir}/{save_name}.txt", "w") as f:
                w, h, d = image.shape
                for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                    f.write(f"{classes_dict[cls]} {x0 / w} {y0 / h} {(x1 - x0) / w} {(y1 - y0) / h}\n")

    @staticmethod
    def create_tfrecord(image_dir: str, annotations_file: str, output_dir) -> Dict:
        pass


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class OIDv4:
    """
    Localisation Annotation Class for the loading and downloading OIDv4 annotations.
    OIDv4 annotations are txt file per image being annotated. For example:

    camera 0.573204 0.619149 0.860499 0.738120
    popcorn 0.758543 0.532122 0.241968 0.665306
    """
    @staticmethod
    def load(image_dir, annotations_dir) -> Tuple[list, list, list, list]:
        """
        Loads a OIDv4 txt files and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: The directory of the annotations file.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise OSError: If there are no image files corresponding to an annotations txt filename.
        :raise OSError: If there more than one image file corresponding to an annotations txt filename.
        """
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]
        names = []
        images = []
        bboxes = []
        classes = []
        for file in annotation_files:
            file_path = f"{annotations_dir}/{file}"
            with open(file_path, "r") as f:
                potential_images = []
                for fmt in IMAGE_FORMATS:
                    image_path = f"{image_dir}/{file.strip('.txt')}{fmt}"
                    if os.path.exists(image_path):
                        potential_images.append(image_path)

                assert len(potential_images) != 0, \
                    f"Theres is no image file in {image_dir} corresponding to the YOLO file {file_path}."
                assert len(potential_images) == 1, \
                    f"Theres are too many image file in {image_dir} corresponding to the YOLO file {file_path}."

                image_path = potential_images[0]
                name = re.split("/|\\\\", image_path)[-1]
                names.append(name)

                image = plt.imread(image_path)
                images.append(images)
                w, h, _ = image.shape

                bboxes_per = []
                classes_per = []
                for line in f.readlines():
                    cls, x0, y0, x1, y1 = line.split()
                    bboxes_per.append(np.asarray([y0, x0, y1, x1], dtype="int64"))
                    classes_per.append(cls)
                bboxes.append(np.asarray(bboxes_per))
                classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a OIDv4 txt files to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f"{download_dir}/{save_name}.txt", "w") as f:
                f.writelines(f"{cls} {x0} {y0} {x1} {y1}\n" for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per))

    @staticmethod
    def create_tfrecord(image_dir: str, annotations_file: str, output_dir) -> Dict:
        pass


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class TensorflowObjectDetectionCSV(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading Tensorflow Object Detection CSV annotations.
    Tensorflow Object Detection CSV annotations is a csv file. For example:

    filename,width,height,class,xmin,ymin,xmax,ymax
    000001.jpg,500,375,helmet,111,144,134,174
    000001.jpg,500,375,helmet,178,84,230,143
    000007.jpg,500,466,helmet,115,139,180,230
    000007.jpg,500,466,helmet,174,156,201,219
    000007.jpg,500,466,helmet,197,177,231,227
    """
    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Loads a Tensorflow Object Detection CSV file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise ValueError: If there is more than one json file in the directory of :param annotations_dir.
        :raise ValueError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith(".csv"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".csv")]
            assert len(potential_annotations) != 0, \
                f"Theres is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"Theres are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]

        image_dict = defaultdict(lambda: {"bboxes": [], "classes": []})
        with open(f"{annotations_dir}/{annotations_file}", "r") as f:
            for row in csv.DictReader(f, delimiter=','):
                name = row["filename"]
                y0, x0, y1, x1 = row["ymin"], row["xmin"], row["ymax"], row["xmax"]
                image_dict[name]["bboxes"].append(np.asarray([y0, x0, y1, x1], dtype="int64"))
                image_dict[name]["classes"].append(row["class"])

        names = []
        images = []
        bboxes = []
        classes = []
        for name, info in image_dict.items():
            names.append(name)
            images.append(plt.imread(f"{image_dir}/{name}"))
            bboxes.append(np.asarray(info["bboxes"]))
            classes.append(np.asarray(info["classes"]))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a Tensorflow Object Detection CSV file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        with open(f"{download_dir}/tensorflow_object_detection_annotations.csv", mode='w') as f:
            fieldnames = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
                w, h, _ = image.shape
                for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                    writer.writerow({
                        "filename": name,
                        "width": w,
                        "height": h,
                        "class": cls,
                        "xmin": x0,
                        "ymin": y0,
                        "xmax": x1,
                        "ymax": y1
                    })

    @staticmethod
    def create_tfrecord(image_dir: str, annotations_file: str, output_dir) -> Dict:
        pass


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class IBMCloud:
    """
    Localisation Annotation Class for the loading and downloading VGG annotations. IBM Cloud uses a .json format.
    For example:

    {
    "version": "1.0",
    "type": "localization",
    "labels": ["Untitled Label", "cat", "mouse"],
    "annotations": {
        "85f760d0-c1b6-4ff4-a7f9-88ee6654a355.jpg": [{
                "x": 0.8056980180806675,
                "y": 0.6729166666666667,
                "x2": 0.8752390472878998,
                "y2": 0.8041666666666667,
                "id": "b523453f-d2bb-4ef1-88bb-1f18cba207c4",
                "label": "mouse"
            }]
        }
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Loads a IBM CLoud json file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise ValueError: If there is more than one json file in the directory of :param annotations_dir.
        :raise ValueError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith(".json"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
            assert len(potential_annotations) != 0, \
                f"Theres is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"Theres are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]
        with open(f"{annotations_dir}/{annotations_file}", "r") as f:
            annotations = json.load(f)

        names = []
        images = []
        bboxes = []
        classes = []
        for filename, annotation in annotations["annotations"].items():
            names.append(filename)
            image = plt.imread(f"{image_dir}/{filename}")
            w, h, _ = image.shape
            images.append(image)

            bboxes_per = []
            classes_per = []
            for a in annotation:
                bboxes_per.append(np.asarray([a["y"] * h, a["x"] * w, a["y2"] * h, a["x2"] * w], dtype="int64"))
                classes_per.append(a["label"])
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a IBM CLoud json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        annotations_info = defaultdict(list)
        annotation_idx = 0
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            w, h, _ = image.shape
            for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                annotations_info[name].append({
                    "label": str(cls),
                    "x": float(x0 / w),
                    "y": float(y0 / h),
                    "x2": float(x1 / w),
                    "y2": float(y1 / h),
                    "id": str(hashlib.md5(str(annotation_idx).encode("utf-8")))
                })
            annotation_idx += 1

        annotations = {
            "version": "1.0",
            "type": "localization",
            "labels": list({str(cls) for classes_per in classes for cls in classes_per}),
            "annotations": annotations_info
        }
        with open(f"{download_dir}/ibm_cloud_annotations.json", "w") as f:
            json.dump(annotations, f)


@strategy_method(LocalisationAnnotationFormats)
@LocalisationAnnotation.register
class VoTTCSV(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading VoTT CSV annotations.
    VoTT CSV annotations is a csv file. For example:

    "image","xmin","ymin","xmax","ymax","label"
    "img0001.jpg",109.02857142857141,86.14285714285714,153.77142857142854,123.94285714285714,"helmet"
    "img0001.jpg",122.69760696156635,18.85103626943005,193.18346627991298,88.48834196891191,"person"
    "img0002.jpg",6.816997518610422,22.483428571428572,195.0452853598015,182.48685714285713,"helmet"
    """
    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Loads a VoTT CSV file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of np.ndarray with the shapes (w, h, d).
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise ValueError: If there is more than one json file in the directory of :param annotations_dir.
        :raise ValueError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith(".csv"):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith(".csv")]
            assert len(potential_annotations) != 0, \
                f"Theres is no annotations .json file in {annotations_dir}."
            assert len(potential_annotations) == 1, \
                f"Theres are too many annotations .json files in {annotations_dir}."
            annotations_file = potential_annotations[0]

        image_dict = defaultdict(lambda: {"bboxes": [], "classes": []})
        with open(f"{annotations_dir}/{annotations_file}", "r") as f:
            for row in csv.DictReader(f, delimiter=','):
                name = row["\"image\""]
                y0, x0, y1, x1 = row["\"ymin\""], row["\"xmin\""], row["\"ymax\""], row["\"xmax\""]
                image_dict[name]["bboxes"].append(np.asarray([y0, x0, y1, x1], dtype="int64"))
                image_dict[name]["classes"].append(row["\"label\""].strip("\""))

        names = []
        images = []
        bboxes = []
        classes = []
        for name, info in image_dict.items():
            name = name.strip("\"")
            names.append(name)
            images.append(plt.imread(f"{image_dir}/{name}"))
            bboxes.append(np.asarray(info["bboxes"]))
            classes.append(np.asarray(info["classes"]))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List[np.ndarray], bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a VoTT CSV file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated. A list of np.ndarray with the shape (width, height, depth).
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise ValueError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            "The params image_names, images bboxes and classes must have the same length." \
            f"len(image_names): {len(image_names)}\n" \
            f"len(images): {len(images)}\n" \
            f"len(bboxes): {len(bboxes)}\n" \
            f"len(classes): {len(classes)}"

        with open(f"{download_dir}/vott_annotations.csv", mode='w') as f:
            fieldnames = ["\"image\"", "\"xmin\"", "\"ymin\"", "\"xmax\"","\"ymax\"", "\"label\""]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
                for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                    writer.writerow({
                        "\"image\"": f"\"{name}\"",
                        "\"label\"": f"\"{cls}\"",
                        "\"xmin\"": x0,
                        "\"ymin\"": y0,
                        "\"xmax\"": x1,
                        "\"ymax\"": y1
                    })

    @staticmethod
    def create_tfrecord(image_dir: str, annotations_file: str, output_dir) -> Dict:
        pass


def convert_annotation_format(image_dir: str, annotations_dir: str, download_dir: str,
                              in_format: Union[LocalisationAnnotation, str],
                              out_format: Union[LocalisationAnnotation, str]) -> None:
    """
    Converts localisation annotation from one format to another.
    :param image_dir: THe directory of where the images are stored.
    :param annotations_dir: The directory of the annotations file.
    :param download_dir: The directory where the annotations are being downloaded.
    :param in_format: The name of the format being converted from.
    :param out_format: THe name of the format being converted to.
    """
    assert isinstance(in_format, (LocalisationAnnotation, str)), \
        f"in_format: {in_format} need to string or LocalisationAnnotation."
    assert isinstance(out_format, (LocalisationAnnotation, str)), \
        f"out_format: {out_format} need to string or LocalisationAnnotation."

    if isinstance(in_format, str):
        in_format = LocalisationAnnotationFormats.get(in_format)

    if isinstance(out_format, str):
        out_format = LocalisationAnnotationFormats.get(out_format)

    out_format.download(download_dir, *in_format.load(image_dir, annotations_dir))

