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

"""Tests for JsonDetector."""

import pytest

import wentral.json_detector as jd


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
