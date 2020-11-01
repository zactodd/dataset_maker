import unittest
from typing import Dict, Type
from abc import ABC, abstractmethod
from dataset_maker import maker


class ImageBBoxTests:
    def setUp(self) -> None:
        self.verification_errors = []


    def tearDown(self) -> None:
        self.assertEqual([], self.verification_errors)

    @property
    @abstractmethod
    def maker(self) -> Type[maker.DatasetMaker]:
        return None

    def test_correct_output_length(self):
        maker = self.maker()
        for o in maker.make(10):
            try:
                self.assertEquals(len(o), 10)
            except AssertionError as e:
                self.verification_errors.append(str(e))

    def test_image_shape_correct_with_defaults(self):
        maker = self.maker()
        for image in maker.make(10)[0]:
            try:
                self.assertEquals(image.shape, (640, 640, 3))
            except AssertionError as e:
                self.verification_errors.append(str(e))

    def test_image_shape_correct(self):
        maker = self.maker(width=120, height=320)
        for image in maker.make(10)[0]:
            try:
                self.assertEquals(image.shape, (120, 320, 3))
            except AssertionError as e:
                self.verification_errors.append(str(e))

    def test_bbox_shape_correct(self):
        maker = self.maker(min_width=50, max_width=100)
        for bboxes_per in maker.make(10)[1]:
            for bbox in bboxes_per:
                try:
                    self.assertEquals(bbox.shape, (4, ))
                except AssertionError as e:
                    self.verification_errors.append(str(e))


class TestSingleSquare(ImageBBoxTests, unittest.TestCase):

    @property
    def maker(self):
        return maker.SingleSquare


class TestMaskSingleSquare(ImageBBoxTests, unittest.TestCase):

    @property
    def maker(self):
        return maker.MaskSingleSquare


class TestMulticlassMultipleSquares(ImageBBoxTests, unittest.TestCase):

    @property
    def maker(self):
        return maker.MulticlassMultipleSquares

