# This file is part of Ad Detect YOLO <https://adblockplus.org/>,
# Copyright (C) 2019-present eyeo GmbH
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

"""Tests for JsonAdDetector."""

import pytest

import ady.json_detector as jd


@pytest.mark.parametrize('image_name,ct,it,expect', [
    ('1.png', None, None, [[10, 10, 80, 25, 0.9], [10, 30, 30, 60, 0.6]]),
    ('1.png', None, 0, [[10, 10, 80, 25, 0.9]]),
    ('0.png', 0.9, None, [[0, 0, 50, 20, 0.9]]),
])
def test_detect(json_output, image_name, ct, it, expect):
    detector = jd.JsonDetector(str(json_output))
    detections = detector.detect(
        None,
        image_name,
        confidence_threshold=ct,
        iou_threshold=it,
    )
    assert detections == expect
