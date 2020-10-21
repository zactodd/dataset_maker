from patterns import SingletonStrategies, strategy_method
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from xml.etree import ElementTree
import matplotlib.pyplot as plt
from functools import reduce
import re
import os


IMAGE_FORMATS = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")


class AnnotationFormats(SingletonStrategies):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Annotations formats: \n" + "\n".join([f"{i:3}: {k}" for i, k in enumerate(self.strategies.keys())])


class Annotation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self, image_dir: str, annotations_file: str) -> Dict:
        """
        Loads in images and annotation files and obtaines relivent adata and puts it in an np array.
        """
        pass

    @abstractmethod
    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        pass



@Annotation.register
class VGG:
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return


@strategy_method(AnnotationFormats)
@Annotation.register
class PascalVOC:
    def load(self, image_dir: str, annotations_dir) -> Tuple[list, list, list, list]:
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
        names = []
        images = []
        bboxes = []
        classes = []
        for f in annotation_files:
            root = ElementTree.parse(f)
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

    def download(self, download_path, image_names, images, bboxes, classes):
        folder = re.split("/|\\\\", download_path)[-1]
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            w, h, d = image.shape

            root = ElementTree.Element("annotation")
            ElementTree.SubElement(root, "folder").text = folder
            ElementTree.SubElement(root, "file").text = name

            size = ElementTree.SubElement(root, "size")
            ElementTree.SubElement(size, "width").text = str(w)
            ElementTree.SubElement(size, "height").text = str(h)
            ElementTree.SubElement(size, "depth").text = str(d)

            for bbox, cls in zip(bboxes_per, classes_per):
                y0, x0, y1, x1 = bbox

                obj = ElementTree.SubElement(root, "object")
                ElementTree.SubElement(obj, "name").text = str(cls)

                bb_elm = ElementTree.SubElement(obj, "bndbox")
                ElementTree.SubElement(bb_elm, "xmin").text = str(x0)
                ElementTree.SubElement(bb_elm, "ymin").text = str(y0)
                ElementTree.SubElement(bb_elm, "xmax").text = str(x1)
                ElementTree.SubElement(bb_elm, "ymax").text = str(y1)

            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f"{download_path}/{save_name}.xml", "wb") as f:
                f.write(ElementTree.tostring(root))


@Annotation.register
class COCO:
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return


@Annotation.register
class YOLO:
    def load(self, image_dir, annotations_dir) -> Tuple[list, list, list, list]:
        pass
                    
    def download(self, download_path, image_names, images, bboxes, classes):
        classes_dict = {n: i for i, n in enumerate({cls for classes_per in classes for cls in classes_per})}
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f"{download_path}/{save_name}.txt", "w") as f:
                w, h, d = image.shape
                for bbox, c in zip(bboxes_per, classes_per):
                    y0, x0, y1, x1 = bbox
                    f.write(f"{classes_dict[c]} {x0 / w} {y0 / h} {(x1 - x0) / w} {(y1 - y0) / h}")


@Annotation.register
class TFRecord:
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return