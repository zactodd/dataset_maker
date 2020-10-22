import unittest
from abc import ABC, abstractmethod
import tempfile
from dataset_maker import annotations as anno
from dataset_maker import maker
import numpy as np
import matplotlib.pyplot as plt


class TestAnnotationBBox:
    def setUp(self):
        dm = maker.MulticlassMultipleSquares(min_width=50, max_width=100)
        self.images, self.bboxes, self.classes = dm.make(10)
        self.names = [f"img_{i}.png" for i in range(len(self.images))]
        self.names_dict = {n: d for n, *d in zip(self.names, self.images, self.bboxes, self.classes)}

        with tempfile.TemporaryDirectory() as td:
            for name, image in zip(self.names, self.images):
                plt.imsave(f"{td}/{name}", image)

            self.annotation_cls.download(td, self.names, self.images, self.bboxes, self.classes)

            self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes = self.annotation_cls.load(td, td)
            self.dl_names_dict = {n: d for n, *d in zip(self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes)}

    @abstractmethod
    def annotation_cls(self):
        return None

    def test_same_names(self):
        self.assertEqual(set(self.names), set(self.dl_names))

    def test_same_bboxes(self):
        for n in self.names:
            bbox = self.names_dict[n][1]
            dl_bbox = self.dl_names_dict[n][1]
            np.testing.assert_equal(bbox, dl_bbox)


class TestPascalVOC(TestAnnotationBBox, unittest.TestCase):
    @property
    def annotation_cls(self):
        return anno.PascalVOC()


class TestYOLO(TestAnnotationBBox, unittest.TestCase):
    @property
    def annotation_cls(self):
        return anno.YOLO()
