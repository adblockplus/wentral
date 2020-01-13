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

"""Tests for visualization."""

from PIL import Image

import ady.visualization as vis


class MatchSetMock:
    """Very basic mock for MatchSet."""

    def __init__(self, true_detections, false_detections,
                 detected_ground_truth, missed_ground_truth):
        self._true_detections = true_detections
        self._false_detections = false_detections
        self._detected_ground_truth = detected_ground_truth
        self._missed_ground_truth = missed_ground_truth


def test_vis_match_set(tmpdir):
    ms = MatchSetMock(
        [(10, 10, 20, 30, 0.55)],
        [(10, 40, 60, 70, 0.45)],
        [(9, 11, 22, 29)],
        [(20, 30, 70, 40), (80, 15, 95, 50)],
    )
    image = Image.new('RGB', (100, 100), (255, 255, 255))
    result = vis.visualize_match_set(ms, image)

    assert result.getpixel((60, 70)) == (255, 255, 255)
    assert result.getpixel((10, 10)) == vis.TD_COLOR
    assert result.getpixel((19, 29)) == vis.TD_COLOR
    assert result.getpixel((10, 40)) == vis.FD_COLOR
    assert result.getpixel((9, 11)) == vis.DGT_COLOR
    assert result.getpixel((10, 11)) == vis.TD_COLOR  # TD is on top of DGT
    assert result.getpixel((20, 30)) == vis.MGT_COLOR
    assert result.getpixel((69, 39)) == vis.MGT_COLOR
