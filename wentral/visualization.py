# This file is part of Wentral
# <https://gitlab.com/eyeo/machine-learning/wentral/>,
# Copyright (C) 2019-present eyeo GmbH
#
# Wentral is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# Wentral is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wentral. If not, see <http://www.gnu.org/licenses/>.

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
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=1)
    if title:
        draw.text((x0 + 2, y0 + 1), title, color)


def _xbox_path(image_name, box, box_type):
    """Make name for the extracted box image file."""
    box = [int(v) for v in box[:4]]
    base_name = os.path.splitext(image_name)[0]
    return '{}_{}_{},{}-{},{}.png'.format(base_name, box_type, *box)


def extract_box(match_set, image, box, box_type, visualizations_path):
    """Extract detection or ground truth image."""
    extracted_path = os.path.join(
        visualizations_path,
        _xbox_path(match_set.image_name, box, box_type),
    )
    image.crop([int(v) for v in box[:4]]).save(extracted_path)


def visualize_match_set(match_set, image, visualizations_path):
    """Draw detection and ground truth boxes according to MatchSet."""
    vis_image = image.copy()

    for dgt in match_set.detected_ground_truth:
        draw_box(vis_image, dgt, DGT_COLOR)
        extract_box(match_set, image, dgt, 'dgt', visualizations_path)
    for mgt in match_set.missed_ground_truth:
        draw_box(vis_image, mgt, MGT_COLOR)
        extract_box(match_set, image, mgt, 'mgt', visualizations_path)
    for td in match_set.true_detections:
        draw_box(vis_image, td, TD_COLOR, '{:.0%}'.format(td[4]))
        extract_box(match_set, image, td, 'td', visualizations_path)
    for fd in match_set.false_detections:
        draw_box(vis_image, fd, FD_COLOR, '{:.0%}'.format(fd[4]))
        extract_box(match_set, image, fd, 'fd', visualizations_path)

    vis_path = os.path.join(visualizations_path, match_set.image_name)
    vis_image.save(vis_path)


def make_summary_dict(evaluation):
    """Produce summary dict for visualization."""
    return [
        {
            'name': ms.image_name,
            'tp': ms.tp,
            'fn': ms.fn,
            'fp': ms.fp,
            'detections': {
                'true': [
                    {
                        'file': _xbox_path(ms.image_name, td, 'td'),
                        'box': td[:5],
                    }
                    for td in ms.true_detections
                ],
                'false': [
                    {
                        'file': _xbox_path(ms.image_name, fd, 'fd'),
                        'box': fd[:5],
                    }
                    for fd in ms.false_detections
                ],
            },
            'ground_truth': {
                'detected': [
                    {
                        'file': _xbox_path(ms.image_name, dgt, 'dgt'),
                        'box': dgt,
                    }
                    for dgt in ms.detected_ground_truth
                ],
                'missed': [
                    {
                        'file': _xbox_path(ms.image_name, mgt, 'mgt'),
                        'box': mgt,
                    }
                    for mgt in ms.missed_ground_truth
                ],
            },
        }
        for ms in evaluation.matchsets
    ]


def write_data_json(evaluation, visualizations_path):
    """Dump evaluation to JSON file for visualization UI."""
    summary = make_summary_dict(evaluation)
    out_path = os.path.join(visualizations_path, 'data.json')
    with open(out_path, 'wt', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def write_index_html(visualizations_path):
    """Write index.html (and supporting code) to visualizations directory."""
    src_dir = os.path.join(os.path.dirname(__file__), 'vis_ui')
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(visualizations_path, file_name)
        shutil.copy(src_path, dst_path)
