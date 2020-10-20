from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import utils


class DatasetMaker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make(self, n_samples: int) -> Any:
        NotImplementedError()
        return "images", "labels"


class SingleSquareBBoxsDM(DatasetMaker):
    def __init__(self, width=640, height=640, min_width=0, max_width=None):
        super().__init__()
        assert max_width is None or min_width < max_width, \
            "min sample width must be smaller than the max sample width."
        assert max_width is None or height > max_width < width, \
            "THe max width must be smaller than the width and height of the image."
        self.width = width
        self.height = height

        self.min_width = min_width
        self.max_width = min(width, height) if max_width is None else max_width

        self._BASE_IMAGE = np.zeros((width, height, 3))

    def make(self, n_samples: int):
        bboxes = []
        images = []
        for _ in range(n_samples):
            image = self._BASE_IMAGE.copy()
            square_width = np.random.randint(self.min_width, self.max_width)

            x0 = np.random.randint(self.width - square_width)
            y0 = np.random.randint(self.height - square_width)
            x1 = x0 + square_width
            y1 = y0 + square_width

            image[x0:x1, y0:y1] = (255, 0, 0)

            bboxes.append(np.asarray([y0, x0, y1, x1]))
            images.append(image)
        return images, bboxes


class MulticlassSingleSquareBBoxsDM(SingleSquareBBoxsDM):
    def __init__(self, n_classes=3, **kwargs):
        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.colours = utils.spec(self.n_classes)

    def make(self, n_samples: int):
        bboxes = []
        images = []
        classes = []
        for _ in range(n_samples):
            image = self._BASE_IMAGE.copy()
            square_width = np.random.randint(self.min_width, self.max_width)

            x0 = np.random.randint(self.width - square_width)
            y0 = np.random.randint(self.height - square_width)
            x1 = x0 + square_width
            y1 = y0 + square_width

            colour = np.random.randint(self.n_classes)
            classes.append(colour)

            image[x0:x1, y0:y1] = self.colours[colour]

            bboxes.append(np.asarray([y0, x0, y1, x1]))
            images.append(image)
        return images, bboxes, classes
