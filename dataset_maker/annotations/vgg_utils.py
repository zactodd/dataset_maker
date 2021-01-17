from typing import Dict, Any
import numpy as np
import warnings


_TWO_PI = 2 * np.pi


def unrecognised_shape_warning(x):
    return warnings.warn(f"{x} is not a recognised, recognised shapes are polygon, polyline, circle, rect and ellipse")


def polygon_dict() -> Dict[str, Any]:
    """
    :return: A dictionary containing base polygon information.
    """
    return {"name": "polygon", "all_points_x": [], "all_points_y": []}


def polyline_to_polygon(polyline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes a polyline into a polygon, assuming the end point connects to the first point.
    :param polyline: A dictionary containing the polyline information.
    :return:
    """
    polyline["name"] = "polygon"
    return polyline


def circle_to_polygon(circle: Dict[str, Any], points: int) -> Dict[str, Any]:
    """
    Makes a polygon approximation of an circle.
    :param circle: A dictionary containing the circle information.
    :param points: The number of points to approximate the circle.
    :return: A polygon approximation of the circle.
    """
    assert points > 0, f"points {points} needs to be greater than zero."
    r, cx, cy = circle["r"], circle["cx"], circle["cy"]
    poly = polygon_dict()
    for t in np.arange(0, _TWO_PI, _TWO_PI / points):
        poly["all_points_x"].append(round(r * np.cos(t) + cx))
        poly["all_points_y"].append(round(r * np.sin(t) + cy))
    return poly


def ellipse_to_polygon(ellipse: Dict[str, Any], points: int) -> Dict[str, Any]:
    """
    Makes a polygon approximation of an ellipse.
    :param ellipse: A dictionary containing the ellipse information.
    :param points: The number of points to approximate the ellipse.
    :return: A polygon approximation of the ellipse.
    """
    assert points > 0, f"points {points} needs to be greater than zero."
    rx, ry, cx, cy, theta = ellipse["rx"], ellipse["ry"], ellipse["cx"], ellipse["cy"], ellipse["theta"]
    poly = polygon_dict()
    for t in np.arange(0, _TWO_PI, _TWO_PI / points):
        x = round(rx * np.cos(t) * np.cos(theta) - ry * np.sin(t) * np.sin(theta) + cx)
        poly["all_points_x"].append(x)
        y = round(rx * np.cos(t) * np.sin(theta) + ry * np.sin(t) * np.cos(theta) + cy)
        poly["all_points_y"].append(y)
    return poly


def rect_to_polygon(rect: Dict[str, Any]) -> Dict[str, Any]:
    """
    Makes a polygon representation of an rectangle (rect VGG dict).
    :param rect: A dictionary containing the rectangle information.
    :return: A polygon representation of an rectangle.
    """
    poly = polygon_dict()
    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
    poly["all_points_x"] = [x, x + w, x + w, x]
    poly["all_points_y"] = [y, y, y + h, y + h]
    return poly


def convert_annotations_to_polygon(annotations: Dict[str, Any], points: int = 32) -> Dict[str, Any]:
    """
    Converts all non polygons in the annotations to polygons.
    :param annotations:
    :param points: The number of points to uses in curved object such as circles and ellipsis
    :return: converted annotations.
    """
    conversions = {
        "polyline": polyline_to_polygon,
        "rect": rect_to_polygon,
        "circle": lambda x: circle_to_polygon(x, points),
        "ellipse": lambda x: ellipse_to_polygon(x, points),
    }
    for image_id, values in annotations.items():
        for i, r in enumerate(values["regions"]):
            shape = r["shape_attributes"]
            if shape:
                name = shape["name"]
                if name in conversions:
                    annotations[image_id]["regions"][i]["shape_attributes"] = conversions[name](shape)
                elif name != "polygon":
                    unrecognised_shape_warning(name)
    return annotations

