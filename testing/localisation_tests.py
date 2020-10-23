import unittest
from abc import abstractmethod
import tempfile
from annoattions import localisation as anno
from dataset_maker import maker
import numpy as np
import matplotlib.pyplot as plt


class TestAnnotation:
    def setUp(self):
        dm = maker.MulticlassMultipleSquares(min_width=50, max_width=100)
        self.images, self.bboxes, self.classes = dm.make(10)
        self.names = [f"img_{i}.png" for i in range(len(self.images))]
        self.names_dict = {n: d for n, *d in zip(self.names, self.images, self.bboxes, self.classes)}

        with tempfile.TemporaryDirectory() as td:
            for name, image in zip(self.names, self.images):
                plt.imsave(f"{td}/{name}", image)

            self.anno_load().download(td, self.names, self.images, self.bboxes, self.classes)

            self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes = self.anno_load().load(td, td)
            self.dl_names_dict = {n: d for n, *d in zip(self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes)}

    @abstractmethod
    def anno_load(self):
        return None

    def test_same_names(self):
        self.assertEqual(set(self.names), set(self.dl_names))

    def test_same_bboxes(self):
        for n in self.names:
            bbox = self.names_dict[n][1]
            dl_bbox = self.dl_names_dict[n][1]
            diff = np.abs(bbox - dl_bbox)
            self.assertTrue(np.all(diff <= 1), msg=f"The bboxs differ by more than one pixel. \n{bbox}\n{dl_bbox}")


class TestPascalVOC(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.PascalVOC()


class TestVGG(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.VGG()


class TestCOCO(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.COCO()


class TestYOLO(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.YOLO()


class TestOIDv4(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.OIDv4()


class TestTensorflowObjectDetectionCSV(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.TensorflowObjectDetectionCSV()


class TestICMCloud(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.IBMCloud()

#
#
# class TestPascalVOCToYOLO(TestAnnotationTranslation, unittest.TestCase):
#     def anno_load(self):
#         return anno.PascalVOC()
#
#     def anno_download(self):
#         return anno.YOLO()
#
#
# class TestPascalVOCToVGG(TestAnnotationTranslation, unittest.TestCase):
#     def anno_load(self):
#         return anno.PascalVOC()
#
#     def anno_download(self):
#         return anno.VGG()
#
#
# class TestYOLOToPascalVOC(TestAnnotationTranslation, unittest.TestCase):
#     def anno_load(self):
#         return anno.YOLO()
#
#     def anno_download(self):
#         return anno.PascalVOC()
#
#
# class TestYOLOToVGG(TestAnnotationTranslation, unittest.TestCase):
#     def anno_load(self):
#         return anno.YOLO()
#
#     def anno_download(self):
#         return anno.VGG()
#
#
# class TestVGGToYOLO(TestAnnotationTranslation, unittest.TestCase):
#     def anno_load(self):
#         return anno.VGG()
#
#     def anno_download(self):
#         return anno.YOLO()
#
#
# class TestVGGToPascalVOC(TestAnnotationTranslation, unittest.TestCase):
#     def anno_load(self):
#         return anno.VGG()
#
#     def anno_download(self):
#         return anno.PascalVOC()

