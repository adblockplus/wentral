# Copyright (C) 2019-present eyeo GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        elif v.kind in {inspect.Parameter.VAR_KEYWORD,
                        inspect.Parameter.VAR_POSITIONAL}:
            # Skip varargs (*args, **kwargs).
            continue
        elif v.default == v.empty:
            t = str(func).split("'")
            func_name = t[1] if len(t) == 3 else func
            raise Exception('Parameter {} is required for detector {}'
                            .format(k, func_name))
    for extra in getattr(args, 'extra', []):
        try:
            name, value = extra.split('=')
        except ValueError:
            raise Exception('Invalid format of extra argument: ' + extra)
        params[name] = value
    return params
