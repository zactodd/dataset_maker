import unittest
from abc import abstractmethod
import tempfile
from dataset_maker.annotations import localisation as anno
from dataset_maker import maker
import matplotlib.pyplot as plt
from dataset_maker.patterns import Singleton
import os
import tensorflow as tf


class TestHelper(metaclass=Singleton):
    def __init__(self):
        self.annotation = None

        self.names = None
        self.images = None
        self.bboxes = None
        self.classes = None
        self.names_dict = None

        self.tf_examples = None

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

                self.annotation.create_tfrecord(td, td, f"{td}/tfrecord", 1)
                tfrecord_files = [f"{td}/{f}" for f in os.listdir(td) if "tfrecord" in f]
                self.tf_examples = list(tf.io.tf_record_iterator(tfrecord_files[0]))


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
        self.tf_examples = th.tf_examples

    def tearDown(self):
        self.assertEqual([], self.verification_errors)

    @abstractmethod
    def anno_load(self):
        return None

    def test_tf_record(self):
        self.assertIsNotNone(self.tf_examples)

    def test_bboxes_y_values(self):
        for example in self.tf_examples:
            result = tf.train.SequenceExample.FromString(example)
            feature = result.context.feature
            try:
                ymins = [float(y.strip("  value:").strip(", "))
                         for y in str(feature["image/object/bbox/ymin"]).split("\n")[1:-2]]
                ymaxs = [float(y.strip("  value:").strip(", "))
                         for y in str(feature["image/object/bbox/ymax"]).split("\n")[1:-2]]
                for y0, y1 in zip(ymins, ymaxs):
                    self.assertGreaterEqual(y1, y0)
            except AssertionError as e:
                self.verification_errors.append(str(e))

    def test_bboxes_x_values(self):
        for example in self.tf_examples:
            result = tf.train.SequenceExample.FromString(example)
            feature = result.context.feature
            try:
                xmins = [float(y.strip("  value:").strip(", "))
                         for y in str(feature["image/object/bbox/xmin"]).split("\n")[1:-2]]
                xmaxs = [float(y.strip("  value:").strip(", "))
                         for y in str(feature["image/object/bbox/xmax"]).split("\n")[1:-2]]
                for x0, x1 in zip(xmins, xmaxs):
                    self.assertGreaterEqual(x1, x0)
            except AssertionError as e:
                self.verification_errors.append(str(e))


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


class TestOIDv4(TestAnnotation, unittest.TestCase):
    def anno_load(self):
        return anno.OIDv4()
