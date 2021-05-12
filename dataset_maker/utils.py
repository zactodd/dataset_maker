import numpy as np
import skimage
# import pycocotools
from typing import Iterable, Tuple
from itertools import islice


def spec(n):
    t = np.linspace(-510, 510, n)
    return np.round(np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)


def bbox(x: Iterable[int], y: Iterable[int], out_format: str = "2p") -> Tuple[int, int, int, int]:
    """
    Gets a bounding boxes from an iterator of x and y points.
    :param x: list of points in which the min and max are selected.
    :param y: list of points in which the min and max are selected.
    :param out_format: The format that the bbox is produced.
    :return: list counting the bounding box.
        2p:
            y0, x0, y1, x1
        width_height:
            y0, y1, height, width
    """
    assert out_format in ("2p", "width_height")
    x0, x1, y0, y1 = min(x), max(x), min(y), max(y)
    if out_format == "2p":
        return y0, x0, y1, x1
    elif out_format == "width_height":
        return y0, x0, y1 - y0, x1 - x0
    else:
        NotImplementedError()


def bbox_area(y0, x0, y1, x1):
    return (y1 - y0) * (x1 - x0)


def polygon_to_mask(x, y, width, height):
    assert len(x) == len(y), f"Length of both x and and y must be the same ({len(x)} != {len(y)})."
    m = np.zeros((width, height), dtype=np.uint8)
    rr, cc = skimage.draw.polygon(y, x)
    m[rr, cc] = 1
    return m


def chunks(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


# TODO Implement workflow to allow pycocotools to be installed
# def rle_to_mask(rle):
#     compressed_rle = pycocotools.mask.frPyObjects(rle, *rle["size"])
#     return pycocotools.mask.decode(compressed_rle)
