import csv

from patterns import SingletonStrategies, strategy_method
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import numpy as np
from xml.etree import ElementTree
import matplotlib.pyplot as plt
from functools import reduce
import utils
import json
import re
import os
from collections import defaultdict


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
            name = root.find("file").text
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
            ElementTree.SubElement(root, "file").text = name

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

        with open(f"{download_dir}/annotations.csv", mode='w') as f:
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


def convert_annotation_format(image_dir: str, annotations_dir: str, download_dir: str, in_format: str, 
                              out_format: str) -> None:
    """
    Converts localisation annotation from one format to another.
    :param image_dir: THe directory of where the images are stored.
    :param annotations_dir: The directory of the annotations file.
    :param download_dir: The directory where the annotations are being downloaded.
    :param in_format: The name of the format being converted from.
    :param out_format: THe name of the format being converted to.
    """
    in_anno = LocalisationAnnotationFormats.get(in_format)
    out_anno = LocalisationAnnotationFormats.get(out_format)
    out_anno.download(download_dir, *in_anno.load(image_dir, annotations_dir))
