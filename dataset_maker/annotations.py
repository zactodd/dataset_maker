from patterns import SingletonStrategies, strategy_method
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from xml.etree import ElementTree
from functools import reduce
import re


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


# def convert_annotation(images_dir: str, annotation_file: str, in_format:str , out_format: str) -> Any:
#     in_ann = AnnotationFormats().get(in_format)
#     out_format = AnnotationFormats().get(out_format)
#     return out_format.download(*in_ann.load(images_dir, annotation_file))


@Annotation.register
class VGG:
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return


@Annotation.register
class PascalVOC:
    def load(self, image_dir: str, annotations_file: str, load_only_annotated=True) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, download_path, image_names, images, bboxes, classes) -> Any:
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

            for bbox, c in zip(bboxes_per, classes_per):
                y0, x0, y1, x1 = bbox

                obj = ElementTree.SubElement(root, "object")
                ElementTree.SubElement(obj, "name").text = str(c)

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
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return


@Annotation.register
class TFRecord:
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return