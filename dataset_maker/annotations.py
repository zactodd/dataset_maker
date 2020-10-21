from patterns import SingletonStrategies, strategy_method
from abc import ABC, abstractmethod
from typing import Tuple, Any
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
    def load(self, image_dir: str, annotations_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        NotImplementedError()
        return

    @abstractmethod
    def download(self, images: np.ndarray, bboxes: np.ndarray, classes: np.ndarray) -> Any:
        NotImplementedError()
        return


def convert_annotation(images_dir: str, annotation_file: str, in_format:str , out_format: str) -> Any:
    in_ann = AnnotationFormats().get(in_format)
    out_format = AnnotationFormats().get(out_format)
    return out_format.download(*in_ann.load(images_dir, annotation_file))
