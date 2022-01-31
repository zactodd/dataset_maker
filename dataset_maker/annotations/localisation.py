import csv
import hashlib
from dataset_maker.patterns import registry
from dataset_maker.annotations.download_upload import LoaderDownloader
from typing import Tuple, List, Union, Dict, Optional
import numpy as np
from xml.etree import ElementTree
from functools import reduce
from dataset_maker import utils
import json
import re
import os
from collections import defaultdict
import tensorflow as tf
from dataset_maker.annotations import dataset_utils, vgg_utils
import contextlib2
from PIL import Image

IMAGE_FORMATS = ('.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')


@registry
class LocalisationAnnotation(LoaderDownloader):
    """
    Abstract base class for LocalisationAnnotation as a LoaderDownloader.
    """


    def __init__(self, name=None, *args, **kwargs) -> None:
        super().__init__()

    def create_tfrecord(self, image_dir: str, annotations_file: str, output_dir: str, num_shards: int = 1,
                        shard_splits: Optional[Tuple[float]] = None, split_names=Optional[Tuple[str]],
                        class_map: Optional[Dict[str, int]] = None) -> None:
        """
        Creates tfrecords from localisation annotations.
        :param image_dir: The image directory.
        :param annotations_file: The location of the annotations or the specific annotations file.
        :param output_dir: THe output directory for the annotations file.
        :param num_shards: THe number of tfrecord files.
        :param shard_splits: The ratio in which the file shards are being split.
        :param split_names: The names of the split shards
        :param class_map: A map of classes to there encoded values by default it will create a map like:
            class_map = {cls: idx for idx, cls in enumerate(unique_classes, 1)}
        """
        filenames, images, bboxes, classes = self.load(image_dir, annotations_file)
        if class_map is None:
            unique_classes = {cls for cls_per in classes for cls in cls_per}
            class_map = {cls: idx for idx, cls in enumerate(unique_classes, 1)}

        with contextlib2.ExitStack() as close_stack:
            if shard_splits is not None:
                assert sum(shard_splits) == 1.0, \
                    f'The sum of shard_splits must equal 1, (sum(shard_splits) = {sum(shard_splits)}).'
                assert split_names is None or (len(shard_splits) == len(split_names)), \
                    f'The length of shard_splits and split_names most be the same ' \
                    f'({len(shard_splits)} != {len(split_names)}).'

                if split_names is None:
                    split_names = [f'split_{i}' for i in range(len(shard_splits))]

                output_tfrecords = dataset_utils.open_sharded_tfrecords_with_splits(close_stack, output_dir, num_shards,
                                                                                    shard_splits, split_names)
            else:
                output_tfrecords = dataset_utils.open_sharded_tfrecords(close_stack, output_dir, num_shards)

            for idx, (filename, image, bbox_per, cls_per) in enumerate(zip(filenames, images, bboxes, classes)):
                # TODO maybe look into different way or find the common standard

                with tf.io.gfile.GFile(f'{image_dir}/{filename}', 'rb') as fid:
                    encoded_image = fid.read()
                width, height = image.size

                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                classes_text = []
                mapped_classes = []

                for (y0, x0, y1, x1), cls in zip(bbox_per, cls_per):
                    ymins.append(float(y0 / height))
                    xmins.append(float(x0 / width))
                    ymaxs.append(float(y1 / height))
                    xmaxs.append(float(x1 / width))
                    classes_text.append(cls.encode('utf8'))
                    mapped_classes.append(class_map[cls])

                image_format = filename.split('.')[-1].encode('utf8')
                encode_filename = filename.encode('utf8')

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': dataset_utils.int64_feature(height),
                    'image/width': dataset_utils.int64_feature(width),
                    'image/filename': dataset_utils.bytes_feature(encode_filename),
                    'image/source_id': dataset_utils.bytes_feature(encode_filename),
                    'image/encoded': dataset_utils.bytes_feature(encoded_image),
                    'image/format': dataset_utils.bytes_feature(image_format),
                    'image/object/bbox/xmin': dataset_utils.float_list_feature(xmins),
                    'image/object/bbox/xmax': dataset_utils.float_list_feature(xmaxs),
                    'image/object/bbox/ymin': dataset_utils.float_list_feature(ymins),
                    'image/object/bbox/ymax': dataset_utils.float_list_feature(ymaxs),
                    'image/object/class/text': dataset_utils.bytes_list_feature(classes_text),
                    'image/object/class/label': dataset_utils.int64_list_feature(mapped_classes)
                }))

                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())


class VGG(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading VGG annotations. VGG Annotation for use a .json
    format. VGG can have its 'regions" as a dictionary with a full VGG file looking like:
    {
        "image_1.png": {
                "regions": {
                        "0": {
                            "shape_attributes": {
                                "name": "polygon",
                                "all_points_x": [0, 25, 25, 0],
                                "all_points_y": [0, 0, 25, 25]
                            },
                            "region_attributes": {"label": "catfish"}
                        }   
        }
    }
    And VGG can have its "regions" as a list with a full VGG file looking like:
    {
        "image_1.png": {
                "regions": [
                                {
                                    "shape_attributes": {
                                        "name": "polygon",
                                        "all_points_x": [0, 25, 25, 0],
                                        "all_points_y": [0, 0, 25, 25]
                                    },
                                    "region_attributes": {"label": "catfish"}
                                }
                ]  
        }
    }
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str, region_label: str = 'label') -> \
            Tuple[List[str], List, List[np.ndarray], List[np.ndarray]]:
        """
        Loads a VGG file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :param region_label: The key that identifies the label being loaded.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the 
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        annotations = utils.open_json_from_file_or_dir(annotations_dir)
        annotations = vgg_utils.convert_annotations_to_polygon(annotations)

        names = []
        images = []
        bboxes = []
        classes = []
        for filename, annotation in annotations.items():
            names.append(filename)
            with Image.open(f'{image_dir}/{filename}') as image:
                images.append(image)

            bboxes_per = []
            classes_per = []

            regions = annotation['regions']
            if isinstance(regions, dict):
                regions = regions.values()

            for r in regions:
                if region_label in r['region_attributes']:
                    bbox = utils.bbox(r['shape_attributes']['all_points_x'], r['shape_attributes']['all_points_y'])
                    bboxes_per.append(np.asarray(bbox))
                    classes_per.append(r['region_attributes'][region_label])
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a VGG json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        annotations = {
            name: {
                'filename': name,
                'regions': {
                    f'{i}': {
                        'shape_attributes': {
                            'name': 'polygon',
                            'all_points_x': [int(x0), int(x1), int(x1), int(x0)],
                            'all_points_y': [int(y0), int(y0), int(y1), int(y1)]
                        },
                        'region_attributes': {'label': str(cls)}
                    }
                    for i, ((y0, x0, y1, x1), cls) in enumerate(zip(bboxes_per, classes_per))
                }
            }
            for name, bboxes_per, classes_per in zip(image_names, bboxes, classes)
        }
        with open(f'{download_dir}/vgg_annotations.json', 'w') as f:
            json.dump(annotations, f)


class PascalVOC(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading Pascal VOC annotations.
    Pascal VOC annotations are xml per image being annotated. For example:
    <annotation>
        <filename>image_1.jpg</filename>
        <size>
            <width>2048</width>
            <height>1536</height>
            <depth>3</depth>
        </size>
        <object>
            <name>dog</name>
            <pose>Unspecified</pose>
            <truncated>Unspecified</truncated>
            <difficult>Unspecified</difficult>
            <bndbox>
                <xmin>293</xmin>
                <ymin>384</ymin>
                <xmax>2055</xmax>
                <ymax>1518</ymax>
            </bndbox>
        </object>
    </annotation>
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> Tuple[list, list, list, list]:
        """
        Loads a Pascal VOC xml files and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: The directory of the annotations file.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        """
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        names = []
        images = []
        bboxes = []
        classes = []
        for f in annotation_files:
            root = ElementTree.parse(f'{annotations_dir}/{f}')
            name = root.find('filename').text
            names.append(name)
            with Image.open(f'{image_dir}/{name}') as image:
                images.append(image)
            bboxes_per = []
            classes_per = []
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                y0 = int(bbox.find('ymin').text)
                x0 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymax').text)
                x1 = int(bbox.find('xmax').text)
                bboxes_per.append(np.asarray([y0, x0, y1, x1]))
                classes_per.append(obj.find('name').text)
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a Pascal VOC xml files to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        folder = re.split('/|\\\\', download_dir)[-1]
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            w, h = image.size
            d = len(image.getbands())

            root = ElementTree.Element('annotation')
            ElementTree.SubElement(root, 'folder').text = folder
            ElementTree.SubElement(root, 'filename').text = name

            size = ElementTree.SubElement(root, 'size')
            ElementTree.SubElement(size, 'width').text = str(w)
            ElementTree.SubElement(size, 'height').text = str(h)
            ElementTree.SubElement(size, 'depth').text = str(d)

            for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                obj = ElementTree.SubElement(root, 'object')
                ElementTree.SubElement(obj, 'name').text = str(cls)

                ElementTree.SubElement(obj, 'pose').text = 'Unspecified'
                ElementTree.SubElement(obj, 'truncated').text = 'Unspecified'
                ElementTree.SubElement(obj, 'difficult').text = 'Unspecified'

                bb_elm = ElementTree.SubElement(obj, 'bndbox')
                ElementTree.SubElement(bb_elm, 'xmin').text = str(x0)
                ElementTree.SubElement(bb_elm, 'ymin').text = str(y0)
                ElementTree.SubElement(bb_elm, 'xmax').text = str(x1)
                ElementTree.SubElement(bb_elm, 'ymax').text = str(y1)

            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f'{download_dir}/{save_name}.xml', 'wb') as f:
                f.write(ElementTree.tostring(root))


class COCO(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading COCO annotations. COCO Annotation for use a .json
    format. For Example:
    {
        "info": {
            "images": [
                {
                    "id": 1,
                    "width": 1504,
                    "height": 2016,
                    "file_name": "image_1.jpg"
                }
            ],
            "annotations": [
                {
                    "id": 0,
                    "iscrowd": 0,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation":[[87.281, 708.408, 1416.71826, 1307.59]],
                    "bbox": [87.281, 708.408, 1416.71826, 1307.59],
                    "area":796574.87632
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "laptop"
                }
            ]
        }
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List, List[np.ndarray], List[np.ndarray]]:
        """
        Loads a COCO file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the 
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        annotations = utils.open_json_from_file_or_dir(annotations_dir)

        classes_dict = {cls_info['id']: cls_info['name'] for cls_info in annotations['categories']}

        image_dict = {
            image_info['id']: {'bboxes': [], 'classes': [], 'name': image_info['file_name']}
            for image_info in annotations['images']
        }
        for annotation in annotations['annotations']:
            idx = annotation['image_id']

            x0, y0, bb_width, bb_height = annotation['bbox']
            x1, y1 = x0 + bb_width, y0 + bb_height

            image_dict[idx]['bboxes'].append(np.asarray([y0, x0, y1, x1], dtype='int64'))
            image_dict[idx]['classes'].append(classes_dict[annotation['category_id']])

        names = []
        images = []
        bboxes = []
        classes = []
        for info in image_dict.values():
            name = info['name']
            names.append(name)
            with Image.open(f'{image_dir}/{name}') as image:
                images.append(image)
            bboxes.append(np.asarray(info['bboxes']))
            classes.append(np.asarray(info['classes']))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a COCO json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        classes_dict = {n: i for i, n in enumerate({cls for classes_per in classes for cls in classes_per}, 1)}
        annotation_idx = 0
        images_info = []
        annotations_info = []
        for img_idx, (name, image, bboxes_per, classes_per) in enumerate(zip(image_names, images, bboxes, classes), 1):
            w, h = image.size
            images_info.append({'id': img_idx, 'file_name': str(name), 'width': int(w), 'height': int(h)})
            for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
                annotations_info.append({
                    'id': annotation_idx,
                    'image_id': img_idx,
                    'category_id': classes_dict[cls],
                    'iscrowd': 0,
                    'segmentation': [bbox],
                    'bbox': bbox,
                    'area': float(utils.bbox_area(y0, x0, y1, x1))
                })
                annotation_idx += 1

        data = {
            'images': images_info,
            'annotations': annotations_info,
            'categories': [{'id': int(cat_idx), 'name': str(cls)} for cls, cat_idx in classes_dict.items()]
        }
        with open(f'{download_dir}/coco_annotations.json', 'w') as f:
            json.dump(data, f)


class YOLO(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading YOLO annotations.
    YOLO annotations are txt file per image being annotated. For example:
    0 0.573204 0.619149 0.860499 0.738120
    1 0.758543 0.532122 0.241968 0.665306
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str, image_format: str = 'png') -> Tuple[list, list, list, list]:
        """
        Loads a YOLO txt files and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: The directory of the annotations file.
        :param image_format: The format of the images being used.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise OSError: If there are no image files corresponding to an annotations txt filename.
        :raise OSError: If there more than one image file corresponding to an annotations txt filename.
        """
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
        names = []
        images = []
        bboxes = []
        classes = []
        for file in annotation_files:
            file_path = f'{annotations_dir}/{file}'
            image_path = f'{image_dir}/{file.strip(".txt")}.{image_format}'
            name = re.split('/|\\\\', image_path)[-1]
            names.append(name)

            with Image.open(image_path) as image:
                w, h = image.size
                images.append(image)

            bboxes_per = []
            classes_per = []

            with open(file_path, 'r') as f:
                for line in f.readlines():
                    cls, *bbox = line.split()
                    x0, y0, dx, dy = [float(p) for p in bbox]
                    bboxes_per.append(np.asarray([y0 * h, x0 * w, (y0 + dy) * h, (x0 + dx) * w], dtype='int64'))
                    classes_per.append(cls)
                bboxes.append(np.asarray(bboxes_per))
                classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a YOLO txt files to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        classes_dict = {n: i for i, n in enumerate({cls for classes_per in classes for cls in classes_per})}
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f'{download_dir}/{save_name}.txt', 'w') as f:
                w, h = image.size
                for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                    f.write(f'{classes_dict[cls]} {x0 / w} {y0 / h} {(x1 - x0) / w} {(y1 - y0) / h}\n')


class OIDv4(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading OIDv4 annotations.
    OIDv4 annotations are txt file per image being annotated. For example:
    camera 0.573204 0.619149 0.860499 0.738120
    popcorn 0.758543 0.532122 0.241968 0.665306
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str, image_format: str = 'png') -> Tuple[list, list, list, list]:
        """
        Loads a OIDv4 txt files and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: The directory of the annotations file.
        :param image_format: The format of the images being used.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        """
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
        names = []
        images = []
        bboxes = []
        classes = []
        for file in annotation_files:
            file_path = f'{annotations_dir}/{file}'
            image_path = f'{image_dir}/{file.strip(".txt")}.{image_format}'
            name = re.split('/|\\\\', image_path)[-1]
            names.append(name)

            with Image.open(image_path) as image:
                images.append(image)

            bboxes_per = []
            classes_per = []

            with open(file_path, 'r') as f:
                for line in f.readlines():
                    cls, x0, y0, x1, y1 = line.split()
                    bboxes_per.append(np.asarray([y0, x0, y1, x1], dtype='int64'))
                    classes_per.append(cls)

            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a OIDv4 txt files to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            save_name = reduce(lambda n, fmt: n.strip(fmt), IMAGE_FORMATS, name)
            with open(f'{download_dir}/{save_name}.txt', 'w') as f:
                f.writelines(f'{cls} {x0} {y0} {x1} {y1}\n' for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per))


class TensorflowObjectDetectionCSV(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading Tensorflow Object Detection CSV annotations.
    Tensorflow Object Detection CSV annotations is a csv file. For example:
    filename,width,height,class,xmin,ymin,xmax,ymax
    000001.jpg,500,375,helmet,111,144,134,174
    000001.jpg,500,375,helmet,178,84,230,143
    000007.jpg,500,466,helmet,115,139,180,230
    000007.jpg,500,466,helmet,174,156,201,219
    000007.jpg,500,466,helmet,197,177,231,227
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List, List[np.ndarray], List[np.ndarray]]:
        """
        Loads a Tensorflow Object Detection CSV file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith('.csv'):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith('.csv')]
            assert len(potential_annotations) != 0, \
                f'There is no annotations .csv file in {annotations_dir}.'
            assert len(potential_annotations) == 1, \
                f'There are too many annotations .csv files in {annotations_dir}.'
            annotations_file = potential_annotations[0]

        image_dict = defaultdict(lambda: {'bboxes': [], 'classes': []})
        with open(f'{annotations_dir}/{annotations_file}', 'r') as f:
            for row in csv.DictReader(f, delimiter=','):
                name = row['filename']
                y0, x0, y1, x1 = row['ymin'], row['xmin'], row['ymax'], row['xmax']
                image_dict[name]['bboxes'].append(np.asarray([y0, x0, y1, x1], dtype='int64'))
                image_dict[name]['classes'].append(row['class'])

        names = []
        images = []
        bboxes = []
        classes = []
        for name, info in image_dict.items():
            names.append(name)
            with Image.open(f'{image_dir}/{name}') as image:
                images.append(image)
            bboxes.append(np.asarray(info['bboxes']))
            classes.append(np.asarray(info['classes']))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a Tensorflow Object Detection CSV file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        with open(f'{download_dir}/tensorflow_object_detection_annotations.csv', mode='w') as f:
            fieldnames = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
                w, h = image.size
                for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                    writer.writerow({
                        'filename': name,
                        'width': w,
                        'height': h,
                        'class': cls,
                        'xmin': x0,
                        'ymin': y0,
                        'xmax': x1,
                        'ymax': y1
                    })


class IBMCloud(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading VGG annotations. IBM Cloud uses a .json format.
    For example:
    {
    "version": "1.0",
    "type": "localization",
    "labels": ["Untitled Label", "cat", "mouse"],
    "annotations": {
        "85f760d0-c1b6-4ff4-a7f9-88ee6654a355.jpg": [{
                "x": 0.8056980180806675,
                "y": 0.6729166666666667,
                "x2": 0.8752390472878998,
                "y2": 0.8041666666666667,
                "id": "b523453f-d2bb-4ef1-88bb-1f18cba207c4",
                "label": "mouse"
            }]
        }
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List, List[np.ndarray], List[np.ndarray]]:
        """
        Loads a IBM CLoud json file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        annotations = utils.open_json_from_file_or_dir(annotations_dir)
        names = []
        images = []
        bboxes = []
        classes = []
        for filename, annotation in annotations['annotations'].items():
            names.append(filename)

            with Image.open(f'{image_dir}/{filename}') as image:
                w, h = image.size
                images.append(image)

            bboxes_per = []
            classes_per = []
            for a in annotation:
                bboxes_per.append(np.asarray([a['y'] * h, a['x'] * w, a['y2'] * h, a['x2'] * w], dtype='int64'))
                classes_per.append(a['label'])
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a IBM CLoud json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        annotations_info = defaultdict(list)
        annotation_idx = 0
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            w, h = image.size
            for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                annotations_info[name].append({
                    'label': str(cls),
                    'x': float(x0 / w),
                    'y': float(y0 / h),
                    'x2': float(x1 / w),
                    'y2': float(y1 / h),
                    'id': str(hashlib.md5(str(annotation_idx).encode('utf-8')))
                })
            annotation_idx += 1

        annotations = {
            'version': '1.0',
            'type': 'localization',
            'labels': list({str(cls) for classes_per in classes for cls in classes_per}),
            'annotations': annotations_info
        }
        with open(f'{download_dir}/ibm_cloud_annotations.json', 'w') as f:
            json.dump(annotations, f)


class VoTTCSV(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading VoTT CSV annotations.
    VoTT CSV annotations is a csv file. For example:
    'image','xmin','ymin','xmax','ymax','label'
    'img0001.jpg',109.02857142857141,86.14285714285714,153.77142857142854,123.94285714285714,'helmet'
    'img0001.jpg',122.69760696156635,18.85103626943005,193.18346627991298,88.48834196891191,'person'
    'img0002.jpg',6.816997518610422,22.483428571428572,195.0452853598015,182.48685714285713,'helmet'
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List, List[np.ndarray], List[np.ndarray]]:
        """
        Loads a VoTT CSV file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        if annotations_dir.endswith('.csv'):
            annotations_file = annotations_dir
        else:
            potential_annotations = [f for f in os.listdir(annotations_dir) if f.endswith('.csv')]
            assert len(potential_annotations) != 0, \
                f'There is no annotations .csv file in {annotations_dir}.'
            assert len(potential_annotations) == 1, \
                f'There are too many annotations .csv files in {annotations_dir}.'
            annotations_file = potential_annotations[0]

        image_dict = defaultdict(lambda: {'bboxes': [], 'classes': []})
        with open(f'{annotations_dir}/{annotations_file}', 'r') as f:
            for row in csv.DictReader(f, delimiter=','):
                name = row['\'image\'']
                y0, x0, y1, x1 = row['\'ymin\''], row['\'xmin\''], row['\'ymax\''], row['\'xmax\'']
                image_dict[name]['bboxes'].append(np.asarray([y0, x0, y1, x1], dtype='int64'))
                image_dict[name]['classes'].append(row['\'label\''].strip('\''))

        names = []
        images = []
        bboxes = []
        classes = []
        for name, info in image_dict.items():
            name = name.strip('\'')
            names.append(name)
            with Image.open(f'{image_dir}/{name}') as image:
                images.append(image)
            bboxes.append(np.asarray(info['bboxes']))
            classes.append(np.asarray(info['classes']))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a VoTT CSV file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        with open(f'{download_dir}/vott_annotations.csv', mode='w') as f:
            fieldnames = ['\'image\'', '\'xmin\'', '\'ymin\'', '\'xmax\'', '\'ymax\'', '\'label\'']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
                for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per):
                    writer.writerow({
                        '\'image\'': f'\'{name}\'',
                        '\'label\'': f'\'{cls}\'',
                        '\'xmin\'': x0,
                        '\'ymin\'': y0,
                        '\'xmax\'': x1,
                        '\'ymax\'': y1
                    })


class CreateML(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading CreateML annotations.
    CreateML annotations is a json file. For example:
    [
        {
            "image": "0001.jpg",
            "annotations": [
                {
                    "label": "tablesaw",
                    "coordinates": {
                        "x": 162.5,
                        "y": 45,
                        "width": 79,
                        "height": 88
                    }
                },
            ]
        }
    ]
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str) -> \
            Tuple[List[str], List, List[np.ndarray], List[np.ndarray]]:
        """
        Loads a CreateML json file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        annotations = utils.open_json_from_file_or_dir(annotations_dir)

        names = []
        images = []
        bboxes = []
        classes = []
        for info in annotations:
            name = info['image']
            names.append(name)
            with Image.open(f'{image_dir}/{name}') as image:
                images.append(image)
            bboxes_per = []
            classes_per = []
            for a in info['annotations']:
                coords = a['coordinates']
                x, y, dy, dx = coords['x'], coords['y'], coords['width'], coords['width']
                bboxes_per.append(np.asarray([y, x, y + dy, x + dx]))
                classes_per.append(a['label'])
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a CreateML json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        annotations = [
            {
                'image': name,
                'annotations': [
                    {
                        'label': str(cls),
                        'coordinates': {
                            'x': float(x0),
                            'y': float(y0),
                            'width': float(x1 - x0),
                            'height': float(y1 - y0)
                        }

                    }
                    for (y0, x0, y1, x1), cls in zip(bbox_per, classes_per)
                ]
            }
            for name, bbox_per, classes_per in zip(image_names, bboxes, classes)
        ]

        with open(f'{download_dir}/vgg_annotations.json', 'w') as f:
            json.dump(annotations, f)


class Remo(LocalisationAnnotation):
    """
    Localisation Annotation Class for the loading and downloading Remo annotations. Remo Annotation for use a .json
    format. For example:
    [
      {
        "file_name": "dog1.jpg",
        "height": 500,
        "width": 750,
        "tags": [],
        "task": "Object detection",
        "annotations": [
          {
            "classes": [
              "Dog"
            ],
            "bbox": {
              "xmin": 339,
              "ymin": 92.5,
              "xmax": 629,
              "ymax": 463.5
            }
          }
        ]
      },
    ]
    """

    @staticmethod
    def load(image_dir: str, annotations_dir: str, region_label: str = 'label') -> \
            Tuple[List[str], List, List[np.ndarray], List[np.ndarray]]:
        """
        Loads a Remo file and gets the names, images bounding boxes and classes for thr image.
        :param image_dir: THe directory of where the images are stored.
        :param annotations_dir: Either a directory of the annotations file or the json annotations file its self.
        :param region_label: The key that identifies the label being loaded.
        :return: Returns names, images bounding boxes and classes
            The names will be a list of strings.
            The images will be a list of PIL images.
            The bounding boxes will be a list of np.ndarray with the shape (n, 4) with the coordinates being the
            format [y0, x0, y1, x1].
            The classes will be a list of of np.ndarray with the shape (n,) and containing string information.
        :raise AssertionError: If there is more than one json file in the directory of :param annotations_dir.
        :raise AssertionError: If there is no json file in the directory of :param annotations_dir.
        """
        annotations = utils.open_json_from_file_or_dir(annotations_dir)

        names = []
        images = []
        bboxes = []
        classes = []
        for annotation in annotations:
            filename = annotation['file_name']
            names.append(filename)
            with Image.open(f'{image_dir}/{filename}') as image:
                images.append(image)

            bboxes_per = []
            classes_per = []
            annotation_info = annotation['annotations']

            for a in annotation_info:
                a_bbox = a['bbox']
                bbox = np.asarray([a_bbox['ymin'], a_bbox['xmin'], a_bbox['ymax'], a_bbox['xmax']])
                for c in a['classes']:
                    bboxes_per.append(bbox)
                    classes_per.append(c)
            bboxes.append(np.asarray(bboxes_per))
            classes.append(np.asarray(classes_per))
        return names, images, bboxes, classes

    @staticmethod
    def download(download_dir: str, image_names: List[str], images: List, bboxes: List[np.ndarray],
                 classes: List[np.ndarray]) -> None:
        """
        Downloads a Remo json file to the :param download_dir with the filename annotations.
        :param download_dir: The directory where the annotations are being downloaded.
        :param image_names: The filenames of the image in the annotations. A list of strings.
        :param images: The images being annotated, A list of PIL images.
        :param bboxes: The bounding boxes to be used as annotations. A list of np.ndarray with the shape (n, 4),
            n being the number of bounding boxes for the image and the bounding boxes in the format [y0, x0, y1, x1].
        :param classes: The classes information for the images. A list of np.ndarray with the shape (n, ),
            n being the number of bounding boxes for the image.
        :raise AssertionError: The length of the params :param image_names, :param images :param bboxes and :param classes
            must be the same.
        """
        assert len(image_names) == len(images) == len(bboxes) == len(classes), \
            'The params image_names, images bboxes and classes must have the same length.' \
            f'len(image_names): {len(image_names)}\n' \
            f'len(images): {len(images)}\n' \
            f'len(bboxes): {len(bboxes)}\n' \
            f'len(classes): {len(classes)}'

        annotations = []
        for name, image, bboxes_per, classes_per in zip(image_names, images, bboxes, classes):
            w, h = image.size
            annotations.append({
                'file_name': name,
                'height': h,
                'width': w,
                'tags': [],
                'task': 'Object detection',
                'annotations': [
                    {
                        'classes': [str(cls)],
                        'bbox': {'xmin': float(x0), 'ymin': float(y0), 'xmax': float(x1), 'ymax': float(y1)}
                    }
                    for (y0, x0, y1, x1), cls in zip(bboxes_per, classes_per)
                ]
            })
        with open(f'{download_dir}/remo_annotations.json', 'w') as f:
            json.dump(annotations, f)


def convert_annotation_tf_record(image_dir: str, annotations_dir: str, download_dir: str,
                                 annotation_format: Union[LocalisationAnnotation, str], num_shard=1,
                                 shard_splits: Optional[Tuple[float]] = None, split_names: Optional[Tuple[str]] = None,
                                 class_map: Optional[Dict[str, int]] = None) -> None:
    """
    Converts localisation annotation format to tf record..
    :param image_dir: THe directory of where the images are stored.
    :param annotations_dir: The directory of the annotations file.
    :param annotation_format: The format of the annotation format being converted.
    :param download_dir: The directory where the annotations are being downloaded.
    :param num_shards: THe number of tfrecord files.
    :param shard_splits: The ratio in which the file shards are being split.
    :param split_names: The names of the split shards
    :param class_map: A map of classes to there encoded values by default it will create a map like:
        class_map = {cls: idx for idx, cls in enumerate(unique_classes, 1)}
    :raise AssertionError: If annotation_format is not string or LocalisationAnnotation.
    """

    assert isinstance(annotation_format, (LocalisationAnnotation, str)), \
        f'in_format: {annotation_format} need to string or LocalisationAnnotation.'

    if isinstance(annotation_format, str):
        annotation_format = LocalisationAnnotation(annotation_format)

    annotation_format.create_tfrecord(image_dir, annotations_dir, download_dir, num_shard, shard_splits,
                                      split_names, class_map)


convert_annotation_format = dataset_utils.annotation_format_converter(LocalisationAnnotation)
