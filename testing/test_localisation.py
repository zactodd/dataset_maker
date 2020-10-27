import unittest
from abc import abstractmethod
import tempfile
from dataset_maker.annotations import localisation as anno
from dataset_maker import maker
import numpy as np
import matplotlib.pyplot as plt
from dataset_maker.patterns import Singleton


class TestHelper(metaclass=Singleton):
    def __init__(self):
        self.annotation = None

        self.names = None
        self.images = None
        self.bboxes = None
        self.classes = None
        self.names_dict = None

        self.dl_names = None
        self.dl_images = None
        self.dl_bboxes = None
        self.dl_classes = None
        self.dl_names_dict = None

        self.tfrecord_examples = None

    def setup_annotation_test(self, annotation):
        if self.annotation != annotation and type(self.annotation) != type(annotation):
            self.annotation = annotation

            dm = maker.MulticlassMultipleSquares(min_width=50, max_width=100)
            self.images, self.bboxes, self.classes = dm.make(10)
            self.names = [f"img_{i}.png" for i in range(len(self.images))]
            self.names_dict = {n: d for n, *d in zip(self.names, self.images, self.bboxes, self.classes)}

            with tempfile.TemporaryDirectory() as td:
                for name, image in zip(self.names, self.images):
                    plt.imsave(f"{td}/{name}", image)

                annotation.download(td, self.names, self.images, self.bboxes, self.classes)

                self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes = self.annotation.load(td, td)
                self.dl_names_dict = {n: d for n, *d in
                                      zip(self.dl_names, self.dl_images, self.dl_bboxes, self.dl_classes)}



class TestAnnotation:
    def setUp(self):
        self.verification_errors = []

        th = TestHelper()
        th.setup_annotation_test(self.anno_load())

        self.names = th.names
        self.images = th.images
        self.bboxes = th.bboxes
        self.classes = th.classes
        self.names_dict = th.names_dict

        self.dl_names = th.dl_names
        self.dl_images = th.dl_images
        self.dl_bboxes = th.bboxes
        self.dl_classes = th.classes
        self.dl_names_dict = th.dl_names_dict

        self.tfrecord_examples = th.tfrecord_examples

    def tearDown(self):
        self.assertEqual([], self.verification_errors)

    @abstractmethod
    def anno_load(self):
        return None

    def test_same_names(self):
        self.assertEqual(set(self.names), set(self.dl_names))

    def test_same_names(self):
        self.assertEqual(sorted(self.names), sorted(self.dl_names))

    def test_same_num_bboxes(self):
        for n in self.names:
            try:
                bbox = self.names_dict[n][1]
                dl_bbox = self.dl_names_dict[n][1]
                self.assertEqual(bbox.shape, dl_bbox.shape)
            except AssertionError as e:
                self.verification_errors.append(str(e))

    def test_same_bboxes(self):
        for n in self.names:
            try:
                bbox = self.names_dict[n][1]
                dl_bbox = self.dl_names_dict[n][1]
                diff = np.abs(bbox - dl_bbox)
                self.assertTrue(np.all(diff <= 1), msg=f"The bboxs differ by more than one pixel. \n{bbox}\n{dl_bbox}")
            except AssertionError as e:
                self.verification_errors.append(str(e))

    def test_same_unique_num_classes(self):
        cls_set = {cls for cls_per in self.classes for cls in cls_per}
        dl_cls_set = {cls for cls_per in self.dl_classes for cls in cls_per}
        self.assertEqual(len(cls_set), len(dl_cls_set))

    def test_same_num_classes(self):
        for n in self.names:
            try:
                classes = self.names_dict[n][2]
                dl_classes = self.dl_names_dict[n][2]
                self.assertEqual(classes.shape, dl_classes.shape)
            except AssertionError as e:
                self.verification_errors.append(str(e))

    def test_tfrecord_give_result(self):
        self.assertIsNotNone(self.tfrecord_examples)


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


class TestTensorflowObjectDetectionCSV(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.TensorflowObjectDetectionCSV()


class TestICMCloud(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.IBMCloud()


class TestVoTTCSV(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.VoTTCSV()


# class TestOIDv4(TestAnnotation, unittest.TestCase):
#     def anno_load(self):
#         return anno.OIDv4()
