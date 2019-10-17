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

"""Tests for the ad detector."""

import os

import PIL
import pytest

import ady.detector as det

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='session')
def ad_detector():
    weights_file = os.environ['YOLOv3_WEIGHTS_PATH']
    return det.YoloAdDetector(weights_file)


def test_detect(ad_detector):
    img_path = os.path.join(DATA_DIR, 'golemde1.png')
    image = PIL.Image.open(img_path)
    boxes = ad_detector.detect(image, img_path)
    assert len(boxes) == 2
    assert boxes[0] == pytest.approx([1109.806911761944, 6.440107639019305,
                                      1181.9722982553335, 602.6969322791466,
                                      0.9956156015396118], 0.1)
    assert boxes[1] == pytest.approx([183.69056995098407, 100.75818575345552,
                                      1025.2654442420373, 402.8758269089919,
                                      0.9925578832626343], 0.1)
