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

"""Tests for common utilities."""

import pytest

import ady.utils as utils


@pytest.mark.parametrize('box1,box2,expect_iou', [
    ((0, 0, 0, 0), (1, 1, 2, 2), 0),          # Zero sized box.
    ((0, 0, 10, 10), (0, 0, 10, 10), 1),      # Same box.
    ((0, 0, 10, 10), (0, 5, 10, 15), 1 / 3),  # Half-overlap.
    ((0, 0, 10, 10), (5, 5, 15, 15), 1 / 7),  # Quarter-overlap.
    ((0, 0, 10, 10), (0, 10, 10, 20), 0),     # Zero area intersection.
    ((0, 0, 10, 10), (0, 11, 10, 21), 0),     # No intersection.
])
def test_iou(box1, box2, expect_iou):
    got_iou = utils.iou(box1, box2)
    assert got_iou == pytest.approx(expect_iou, 0.001)


def test_bounding_box():
    assert utils.bounding_box(
        (10, 20, 30, 40),
        (5, 50, 15, 60),
        (20, 30, 40, 50),
    ) == (5, 20, 40, 60)
