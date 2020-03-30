# This file is part of Ad Detect YOLO <https://adblockplus.org/>,
# Copyright (C) 2019 eyeo GmbH
#
# Ad Detect YOLO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# Ad Detect YOLO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ad Detect YOLO. If not, see <http://www.gnu.org/licenses/>.

"""Common utilities."""

import inspect

# Add this to a possibly zero-valued denominator to avoid division by zero.
EPSILON = 1e-7


def area(box):
    """Calculate area of a box."""
    if box is None:
        return 0
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0)


def intersect(box1, box2):
    """Calculate the intersection of two boxes."""
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    x0 = max(b1_x0, b2_x0)
    y0 = max(b1_y0, b2_y0)
    x1 = min(b1_x1, b2_x1)
    y1 = min(b1_y1, b2_y1)

    if x0 > x1 or y0 > y1:  # No intersection, return None
        return None

    return (x0, y0, x1, y1)


def bounding_box(*boxes):
    """Compute a bounding box around other boxes."""
    x0, y0, x1, y1 = boxes[0]

    for bx0, by0, bx1, by1 in boxes[1:]:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)

    return x0, y0, x1, y1


def iou(box1, box2):
    """Calculate intersection over union of two boxes."""
    int_area = area(intersect(box1, box2))
    return int_area / (area(box1) + area(box2) - int_area + EPSILON)


def xy_swap(box):
    """Swap x and y coordinates in a box."""
    x0, y0, x1, y1 = box[:4]
    return (y0, x0, y1, x1) + box[4:]


def kwargs_from_ns(func, args):
    """Extract kwargs for calling `func` from argparse namespace."""
    signature = inspect.signature(func)
    params = {}
    for k, v in signature.parameters.items():
        if getattr(args, k, None) is not None:
            params[k] = getattr(args, k)
        elif v.default == v.empty:
            raise Exception('Parameter {} is required for detector {}'
                            .format(k, args.detector))
    for extra in getattr(args, 'extra', []):
        try:
            name, value = extra.split('=')
        except ValueError:
            raise Exception('Invalid format of extra argument: ' + extra)
        params[name] = value
    return params
