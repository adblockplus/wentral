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

"""Visualization of detections, ground truth and their matching."""

from PIL import ImageDraw


# Colors for different kinds of boxes.
TD_COLOR = (0, 200, 0)    # true detection = green.
FD_COLOR = (200, 100, 0)  # false detection = orange.
DGT_COLOR = (0, 0, 255)   # detected ground truth = blue.
MGT_COLOR = (255, 0, 0)   # missed ground truth = red.


def draw_box(image, box, color, title=None):
    """Draw a box on the image."""
    x0, y0, x1, y1 = box[:4]
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=1)
    if title:
        draw.text((x0 + 2, y0 + 1), title, color)


def visualize_match_set(match_set, image):
    """Draw detection and ground truth boxes according to MatchSet."""
    image = image.copy()

    for dgt in match_set._detected_ground_truth:
        draw_box(image, dgt, DGT_COLOR)
    for mgt in match_set._missed_ground_truth:
        draw_box(image, mgt, MGT_COLOR)
    for td in match_set._true_detections:
        draw_box(image, td, TD_COLOR, '{:.0%}'.format(td[4]))
    for fd in match_set._false_detections:
        draw_box(image, fd, FD_COLOR, '{:.0%}'.format(fd[4]))

    return image
