import numpy as np
from typing import Iterable, Tuple


def spec(n):
    t = np.linspace(-510, 510, n)
    return np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)


def bbox(x: Iterable[int], y: Iterable[int], out_format: str = "2p") -> Tuple[int, int, int, int]:
    """
    Gets a bounding boxes from an iterator of x and y points.
    :param x: list of points in which the min and max are selected.
    :param y: list of points in which the min and max are selected.
    :param out_format: The format that the bbox is produced.
    :return: list counting the bounding box.
        2p: x0, y0, x1, y1
        width_height: x0, y0, width, height
    """
    assert out_format in ("2p", "width_height")
    x0, x1, y0, y1 = min(x), max(x), min(y), max(y)
    if out_format == "2p":
        return y0, x0, y1, x1
    elif out_format == "width_height":
        return y0, x0, y1 - y0, x1 - x0
    else:
        NotImplementedError()
