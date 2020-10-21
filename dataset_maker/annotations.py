from patterns import SingletonStrategies, strategy_method
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


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


def convert_annotation(images_dir: str, annotation_file: str, in_format:str , out_format: str) -> Any:
    in_ann = AnnotationFormats().get(in_format)
    out_format = AnnotationFormats().get(out_format)
    return out_format.download(*in_ann.load(images_dir, annotation_file))


@Annotation.register
class VGG:
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return


@Annotation.register
class PascalVOC:
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return

    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        return


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
