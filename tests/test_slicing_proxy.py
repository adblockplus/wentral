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

"""Tests for slicing detection proxy."""

from PIL import Image
import pytest
from unittest import mock

import wentral.slicing_detector_proxy as sdp
import wentral.utils as utils

import conftest


@pytest.mark.parametrize('size,threshold,overlap,expect', [
    ((20, 22), 0.7, 0.2, [(0, 0, 20, 22)]),
    ((22, 20), 0.7, 0.2, [(0, 0, 22, 20)]),
    ((50, 20), 0.3, 0.2, [(0, 0, 50, 20)]),
    ((50, 20), 0.7, 0.2, [(0, 0, 20, 20), (15, 0, 35, 20), (30, 0, 50, 20)]),
    ((20, 50), 0.7, 0.2, [(0, 0, 20, 20), (0, 15, 20, 35), (0, 30, 20, 50)]),
    ((50, 20), 0.7, 0.6, [(0, 0, 20, 20), (7, 0, 27, 20), (15, 0, 35, 20),
                          (22, 0, 42, 20), (30, 0, 50, 20)]),
    ((1000, 10), 0.7, 0.2, None),
    ((10, 1000), 0.7, 0.9, None),
])
def test_slice_boxes(size, threshold, overlap, expect):
    """Test slice box generation."""
    got = sdp.SlicingDetectorProxy._slice_boxes(size, threshold, overlap)
    # Check the result matches the expectation (if we have one).
    if expect is not None:
        assert got == expect
    # Check squareness.
    for x0, y0, x1, y1 in got:
        width = x1 - x0
        height = y1 - y0
        assert (width * threshold <= height and
                height * threshold <= width)
    # Check overlap.
    for (s1, s2) in zip(got, got[1:]):
        assert utils.area(utils.intersect(s1, s2)) >= utils.area(s1) * overlap
    # Check start and end.
    assert got[0][0:2] == (0, 0)
    assert got[-1][2:4] == size


# Boxes and detections for combining tests.
BOX1 = (0, 0, 20, 20)
D1 = (5, 5, 15, 20, 0.8)
D11 = (0, 0, 5, 5, 0.8)
D12 = (3, 13, 8, 20, 0.9)

BOX2 = (0, 10, 20, 30)
D2 = (4, 13, 14, 23, 0.6)
D21 = (4, 13, 10, 23, 0.4)  # D2 separated into
D22 = (8, 13, 14, 22, 0.3)  # two parts.
D23 = (0, 25, 5, 30, 0.8)

BOX3 = (0, 20, 20, 40)
BOX31 = (0, 21, 20, 40)
D3 = (5, 25, 15, 35, 0.8)

D1_2 = (4, 5, 15, 23, 0.8)     # D1 + D2 = D1 + D21 + D22
D1_22 = (5, 5, 15, 22, 0.8)    # D1 + D22
D1_1221 = (3, 5, 15, 23, 0.9)  # D1 + D12 + D21 + D22


@pytest.mark.parametrize('box1,dets1,box2,dets2,iou_threshold,expected', [
    # No detections.
    (BOX1, [], BOX2, [], 0.5, []),
    # Non-overlapping boxes.
    (BOX1, [D1], BOX31, [D3], 0.5, [D1, D3]),
    # Overlap over a zero-area rectangle.
    (BOX1, [D1], BOX3, [D3], 0.5, [D1, D3]),
    # Overlapping boxes with detections that don't overlap.
    (BOX1, [D11], BOX2, [D23], 0.5, [D11, D23]),
    # Two overlapping detections.
    (BOX1, [D1], BOX2, [D2], 0.5, [D1_2]),
    # One overlapping two others.
    (BOX1, [D1], BOX2, [D21, D22], 0.3, [D1_2]),
    # a overlaps b overlaps c overlaps d + e, f that don't overlap anything.
    (BOX1, [D1, D11, D12], BOX2, [D21, D22, D23], 0.3, [D1_1221, D11, D23]),
])
def test_combine_box_detections(box1, dets1, box2, dets2, iou_threshold,
                                expected):
    """Test combining detections from two boxes."""
    got1 = sdp.SlicingDetectorProxy._combine_box_detections(
        box1, dets1,
        box2, dets2,
        iou_threshold,
    )
    assert set(got1) == set(expected)

    # Also check if it works in reverse order.
    got2 = sdp.SlicingDetectorProxy._combine_box_detections(
        box2, dets2,
        box1, dets1,
        iou_threshold,
    )
    assert set(got2) == set(expected)

    # And now swap x and y and check that is still works.
    s_box1 = utils.xy_swap(box1)
    s_box2 = utils.xy_swap(box2)
    s_dets1 = list(map(utils.xy_swap, dets1))
    s_dets2 = list(map(utils.xy_swap, dets2))
    s_expected = list(map(utils.xy_swap, expected))
    s_got1 = sdp.SlicingDetectorProxy._combine_box_detections(
        s_box1, s_dets1,
        s_box2, s_dets2,
        iou_threshold,
    )
    assert s_got1 == s_expected


def make_relative(det, box):
    """Make detections relative to the box.

    Used for the parameters of test_combine_slice_detections.
    """
    x0, y0, x1, y1, p = det
    bx0, by0, *_ = box
    return x0 - bx0, y0 - by0, x1 - bx0, y1 - by0, p


# Detections from BOX2 and BOX3 made relative to the boxes (that's how they
# come out of the underlying detector).
R_D21 = make_relative(D21, BOX2)
R_D22 = make_relative(D22, BOX2)
R_D3 = make_relative(D3, BOX3)


@pytest.mark.parametrize('slice_detections,iou_threshold,expect', [
    # Low threshold: everything is merged.
    ([[D1, D11], [R_D21, R_D22], [R_D3]], 0.3, [D11, D1_2, D3]),
    # Medium threshold: D1 is merged with D22 but not D21.
    ([[D1, D11], [R_D21, R_D22], [R_D3]], 0.4, [D11, D1_22, D21, D3]),
    # High threshold: nothing is merged.
    ([[D1, D11], [R_D21, R_D22], [R_D3]], 0.8, [D1, D11, D21, D22, D3]),
])
def test_combine_slice_detections(slice_detections, iou_threshold, expect):
    """Test combining detections from multiple slices."""
    slice_boxes = [BOX1, BOX2, BOX3]
    got = sdp.SlicingDetectorProxy._combine_slice_detections(
        slice_boxes, slice_detections, iou_threshold,
    )
    assert set(got) == set(expect)

    # Flip x and y and check that it still works.
    s_slice_boxes = list(map(utils.xy_swap, slice_boxes))
    s_slice_detections = [
        list(map(utils.xy_swap, dets))
        for dets in slice_detections
    ]
    s_expect = list(map(utils.xy_swap, expect))
    s_got = sdp.SlicingDetectorProxy._combine_slice_detections(
        s_slice_boxes, s_slice_detections, iou_threshold,
    )
    assert set(s_got) == set(s_expect)


@pytest.mark.parametrize('slice_detections,iou_threshold,expect', [
    ([[D1, D11], [R_D21, R_D22], [R_D3]], 0.3, [D11, D1_2, D3]),
])
def test_slicing_detection_proxy(slice_detections, iou_threshold, expect):
    """End-to-end test of the slicing detector proxy."""
    slice_boxes = [BOX1, BOX2, BOX3]
    image = Image.new('RGB', (20, 40), (0, 0, 0))
    img_name = 'foo'
    frag_path_tmpl = '{0}_{1[0]},{1[1]}-{1[2]},{1[3]}'
    detector = conftest.MockDetector({
        frag_path_tmpl.format(img_name, box): dets
        for box, dets in zip(slice_boxes, slice_detections)
    })
    proxy = sdp.SlicingDetectorProxy(detector, iou_threshold,
                                     slice_overlap=0.5)
    got = proxy.detect(image, 'foo')
    assert set(expect) == set(got)


def test_slicing_detection_proxy_argument_passthrough():
    """Test that arguments of `detect()` are passed through correctly:

    - iou_threshold should not be passed through (SlicingDetectorProxy will
      use it's default of 0.3 inside, but the wrapped detector might have
      another default).
    - confidence_threshold should not be passed as it's given as an argument.
    - slicing_threshold should not be passed, because it's the parameter of
      SlicingDetectorProxy.
    - extra_arg should be passed.

    """
    image = Image.new('RGB', (20, 30), (0, 0, 0))
    detector = mock.MagicMock()
    detector.batch_detect.return_value = [
        ('foo_0,0-20,20', []),
        ('foo_0,10-20,30', []),
    ]
    proxy = sdp.SlicingDetectorProxy(detector, 0.3, slice_overlap=0.5)
    proxy.detect(image, 'foo', 0.1, slicing_threshold=0.9, extra_arg=42)
    assert len(detector.mock_calls) == 1
    name, _, kwargs = detector.mock_calls[0]
    assert name == 'batch_detect'
    assert kwargs == {'confidence_threshold': 0.1, 'extra_arg': 42}


def test_combine_order():
    """Detections are correctly combined regardless of order.

    `batch_detect()` doesn't have to return the detections in the same order
    that the images came in. Nevertheless, we should combine them right.

    """
    image = Image.new('RGB', (20, 30), (0, 0, 0))
    detector = mock.MagicMock()
    detector.batch_detect.return_value = [
        ('foo_0,10-20,30', [(15, 15, 16, 16, 0.15)]),
        ('foo_0,0-20,20', [(0, 0, 1, 1, 0.1)]),
    ]
    proxy = sdp.SlicingDetectorProxy(detector, 0.3, slice_overlap=0.5)
    got = proxy.detect(image, 'foo')
    assert set(got) == {(0, 0, 1, 1, 0.1), (15, 25, 16, 26, 0.15)}
