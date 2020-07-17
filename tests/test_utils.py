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

"""Tests for common utilities."""

import pytest

import wentral.utils as utils


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
