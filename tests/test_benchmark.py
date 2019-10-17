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

"""Tests for the benchmarking code."""

import pytest

import ady.benchmark as bm


@pytest.fixture()
def dataset(dataset_dir):
    """Dataset wrapper over prepared images and regions."""
    return bm.LabeledDataset(str(dataset_dir))


def assert_match_set(ms, tp, fn, fp, precision, recall):
    assert ms.tp == tp
    assert ms.fn == fn
    assert ms.fp == fp
    assert precision - 0.001 < ms.precision < precision + 0.001
    assert recall - 0.001 < ms.recall < recall + 0.001


def test_match_detections(dataset, mock_detector):
    results = list(bm.match_detections(dataset, mock_detector))
    assert_match_set(results[0], 2, 0, 1, 0.666, 1)
    assert_match_set(results[1], 1, 0, 1, 0.5, 1)
    assert_match_set(results[2], 1, 2, 2, 0.333, 0.333)


def test_evaluate(dataset, mock_detector):
    result = bm.evaluate(dataset, mock_detector)
    assert_match_set(result, 4, 2, 4, 0.5, 4 / 6)


@pytest.mark.parametrize('iou,expect', [
    # Strict IoU, imprecise detections are not counted.
    (0.95, (3, 3, 5, 3 / 8, 3 / 6)),
    # Loose IoU, even very imprecise detections count.
    (0.1, (6, 0, 2, 6 / 8, 1)),
])
def test_evaluate_iou(dataset, mock_detector, iou, expect):
    result = bm.evaluate(dataset, mock_detector, iou)
    assert_match_set(result, *expect)
