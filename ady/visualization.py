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

import json
import os
import shutil

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

    for dgt in match_set.detected_ground_truth:
        draw_box(image, dgt, DGT_COLOR)
    for mgt in match_set.missed_ground_truth:
        draw_box(image, mgt, MGT_COLOR)
    for td in match_set.true_detections:
        draw_box(image, td, TD_COLOR, '{:.0%}'.format(td[4]))
    for fd in match_set.false_detections:
        draw_box(image, fd, FD_COLOR, '{:.0%}'.format(fd[4]))

    return image


def make_summary_dict(evaluation):
    """Produce summary dict for visualization."""
    return [
        {
            'name': ms.image_name,
            'tp': ms.tp,
            'fn': ms.fn,
            'fp': ms.fp,
        }
        for ms in evaluation.matchsets
    ]


def write_data_js(evaluation, visualizations_path):
    """Dump evaluation to JS data file for visualization UI."""
    summary = make_summary_dict(evaluation)
    out_path = os.path.join(visualizations_path, 'data.js')
    with open(out_path, 'wt', encoding='utf-8') as f:
        f.write('imageData = ')
        json.dump(summary, f, indent=2)
        f.write(';')


def write_index_html(visualizations_path):
    """Write index.html to the visualizations directory."""
    src_dir = os.path.join(os.path.dirname(__file__), 'vis_ui')
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(visualizations_path, file_name)
        shutil.copy(src_path, dst_path)
