import unittest
from abc import ABC, abstractmethod
import tempfile
from dataset_maker import annotations as anno
from dataset_maker import maker
import numpy as np
import matplotlib.pyplot as plt


class TestAnnotationTranslation:
    def setUp(self):
        dm = maker.MulticlassMultipleSquares(min_width=50, max_width=100)
        self.images, self.bboxes, self.classes = dm.make(10)
        self.names = [f"img_{i}.png" for i in range(len(self.images))]
        self.names_dict = {n: d for n, *d in zip(self.names, self.images, self.bboxes, self.classes)}

        with tempfile.TemporaryDirectory() as td:
            for name, image in zip(self.names, self.images):
                plt.imsave(f"{td}/{name}", image)

            self.anno_download().download(td, self.names, self.images, self.bboxes, self.classes)

            self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes = self.anno_load().load(td, td)
            self.dl_names_dict = {n: d for n, *d in zip(self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes)}

    @abstractmethod
    def anno_load(self):
        return None

    @abstractmethod
    def anno_download(self):
        return None

    def test_same_names(self):
        self.assertEqual(set(self.names), set(self.dl_names))

    def test_same_bboxes(self):
        for n in self.names:
            bbox = self.names_dict[n][1]
            dl_bbox = self.dl_names_dict[n][1]
            np.testing.assert_equal(bbox, dl_bbox)


class TestAnnotationSelfTranslation(TestAnnotationTranslation):
    @abstractmethod
    def anno_load(self):
        return None

    def anno_download(self):
        return self.anno_load()


class TestPascalVOC(TestAnnotationSelfTranslation, unittest.TestCase):
    def anno_load(self):
        return anno.PascalVOC()


class TestYOLO(TestAnnotationSelfTranslation, unittest.TestCase):
    def anno_load(self):
        return anno.YOLO()


class TestPascalVOCToYOLO(TestAnnotationTranslation, unittest.TestCase):
    def anno_load(self):
        return anno.PascalVOC()

    def anno_download(self):
        return anno.YOLO()


class TestYOLOToPascalVOC(TestAnnotationTranslation, unittest.TestCase):
    def anno_load(self):
        return anno.YOLO()

    def anno_download(self):
        return anno.PascalVOC()
