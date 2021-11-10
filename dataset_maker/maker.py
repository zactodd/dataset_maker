from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from dataset_maker import utils


class DatasetMaker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make(self, n_samples: int) -> Any:
        NotImplementedError()
        return


class SingleSquare(DatasetMaker):
    def __init__(self, width=640, height=640, min_width=0, max_width=None):
        super().__init__()
        assert max_width is None or min_width < max_width, \
            'min sample width must be smaller than the max sample width.'
        assert max_width is None or height > max_width < width, \
            'THe max width must be smaller than the width and height of the image.'
        self.width = width
        self.height = height

        self.min_width = min_width
        self.max_width = min(width, height) if max_width is None else max_width

        self._BASE_IMAGE = np.zeros((width, height, 3))

    def make(self, n_samples: int):
        images = []
        bboxes = []
        for _ in range(n_samples):
            image = self._BASE_IMAGE.copy()
            bbox = y0, x0, y1, x1 = self.rand_square()
            image[x0:x1, y0:y1] = (255, 0, 0)
            images.append(image)
            bboxes.append([np.asarray(bbox)])
        return images, bboxes

    def rand_square(self):
        square_width = np.random.randint(self.min_width, self.max_width)
        x0 = np.random.randint(self.width - square_width)
        y0 = np.random.randint(self.height - square_width)
        return y0, x0, y0 + square_width, x0 + square_width


class MaskSingleSquare(SingleSquare):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def make(self, n_samples: int):
        images = []
        bboxes = []
        masks = []
        for _ in range(n_samples):
            image = self._BASE_IMAGE.copy()
            bbox = y0, x0, y1, x1 = self.rand_square()
            image[x0:x1, y0:y1] = (1, 0, 0)
            images.append(image)
            bboxes.append(np.asarray([bbox]))
            masks.append((image == (1, 0, 0)).all(axis=2))
        return images, bboxes, masks


class MulticlassSingleSquare(SingleSquare):
    def __init__(self, n_classes=3, **kwargs):
        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.colours = utils.spec(self.n_classes)

    def make(self, n_samples: int):
        images = []
        bboxes = []
        classes = []
        for _ in range(n_samples):
            image = self._BASE_IMAGE.copy()
            bbox = y0, x0, y1, x1 = self.rand_square()

            colour = np.random.randint(self.n_classes)
            classes.append(colour)

            image[x0:x1, y0:y1] = self.colours[colour]

            bboxes.append(np.asarray(bbox))
            images.append(image)
        return images, bboxes, classes


class MultipleSquares(SingleSquare):
    def __init__(self, min_n_per_image=1, max_n_per_image=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert min_n_per_image <= max_n_per_image, \
            'Min examples per image must be less or equal than the max examples per image'
        self.min_n_per_image = min_n_per_image
        self.max_n_per_image = max_n_per_image

    def make(self, n_samples: int):
        images = []
        bboxes = []
        for _ in range(n_samples):
            image = self._BASE_IMAGE.copy()

            image_bboxes = []
            if self.min_n_per_image == self.max_n_per_image:
                n_examples = self.min_n_per_image
            else:
                n_examples = np.random.randint(self.min_n_per_image, self.max_n_per_image)

            for _ in range(n_examples):
                bbox = y0, x0, y1, x1 = self.rand_square()
                image[x0:x1, y0:y1] = (1, 0, 0)
                image_bboxes.append(np.asarray(bbox))
            images.append(image)
            bboxes.append(np.asarray(image_bboxes))
        return images, bboxes


class MulticlassMultipleSquares(MultipleSquares):
    def __init__(self, n_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.colours = utils.spec(self.n_classes)

    def make(self, n_samples: int):
        images = []
        bboxes = []
        classes = []
        for _ in range(n_samples):
            image = self._BASE_IMAGE.copy()

            if self.min_n_per_image == self.max_n_per_image:
                n_examples = self.min_n_per_image
            else:
                n_examples = np.random.randint(self.min_n_per_image, self.max_n_per_image)

            image_bboxes = []
            image_classes = []
            for _ in range(n_examples):
                bbox = y0, x0, y1, x1 = self.rand_square()
                colour = np.random.randint(self.n_classes)
                image[x0:x1, y0:y1] = self.colours[colour] / 255

                image_bboxes.append(np.asarray(bbox))
                image_classes.append(colour)
            images.append(image)
            bboxes.append(np.asarray(image_bboxes))
            classes.append(np.asarray(image_classes))
        return images, bboxes, classes
