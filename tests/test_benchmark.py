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


# Data for PRC, IPRC and mAP tests.
DETECTIONS = [
    (0.9, True),
    (0.8, True),
    (0.7, False),
    (0.6, False),
    (0.6, False),
    (0.5, True),
    (0.4, True),
    (0.3, False),
    (0.2, False),
    (0.1, True),
]

GROUND_TRUTH = [1, 2, 3, 4, 5]  # We only care about the length of this list.

EXPECT_PRC = [
    (1.0, 1.0, 0),
    (0.9, 1.0, 0.2),
    (0.8, 1.0, 0.4),
    (0.7, 0.67, 0.4),
    (0.6, 0.4, 0.4),
    (0.5, 0.5, 0.6),
    (0.4, 0.57, 0.8),
    (0.3, 0.5, 0.8),
    (0.2, 0.44, 0.8),
    (0.1, 0.5, 1.0),
    (0.0, 0, 1),
]

EXPECT_IPRC = [
    (1.0, 1.0, 0),
    (0.9, 1.0, 0.2),
    (0.8, 1.0, 0.4),
    (0.7, 0.67, 0.4),
    (0.6, 0.57, 0.4),
    (0.5, 0.57, 0.6),
    (0.4, 0.57, 0.8),
    (0.3, 0.5, 0.8),
    (0.2, 0.5, 0.8),
    (0.1, 0.5, 1.0),
    (0.0, 0, 1),
]


def test_prc():
    """Test precision-recall curve computation."""
    prc = bm._precision_recall_curve(DETECTIONS, GROUND_TRUTH)
    assert len(prc) == len(EXPECT_PRC)
    for p, e in zip(prc, EXPECT_PRC):
        assert p == pytest.approx(e, 0.1)


def test_iprc():
    """Test precision-recall curve interpolation."""
    iprc = bm._interpolate_prc(EXPECT_PRC)
    assert len(iprc) == len(EXPECT_IPRC)
    for p, e in zip(iprc, EXPECT_IPRC):
        assert p == pytest.approx(e, 0.1)


def test_auc():
    """Test AUC computation."""
    assert bm._auc(EXPECT_PRC) == pytest.approx(0.691, 0.001)


def test_ap():
    """Test average precision computation."""
    ap = bm.average_precision(DETECTIONS, GROUND_TRUTH)
    assert ap == pytest.approx(0.728, 0.001)


@pytest.mark.parametrize('detections,ground_truth,expect_ap', [
    ([(0.999, True), (0.999, True)], [1, 2], 1),     # All detected.
    ([(0.999, True), (0.999, False)], [1, 2], 0.5),  # Half detected.
    ([(0.999, False), (0.999, False)], [1, 2], 0),   # None detected.
])
def test_ap_all_1(detections, ground_truth, expect_ap):
    """Test AP with everything detected at c=1.0 (happens with adybm -r)."""
    ap = bm.average_precision(detections, ground_truth)
    assert ap == expect_ap


def assert_match_set(ms, tp, fn, fp, precision, recall, mAP=None):
    assert ms.tp == tp
    assert ms.fn == fn
    assert ms.fp == fp
    assert ms.precision == pytest.approx(precision, 0.01)
    assert ms.recall == pytest.approx(recall, 0.01)
    if mAP is not None:
        assert ms.mAP == pytest.approx(mAP, 0.01)


def test_match_detections(dataset, mock_detector):
    results = list(bm.match_detections(dataset, mock_detector))
    assert_match_set(results[0], 2, 0, 1, 0.666, 1)
    assert_match_set(results[1], 1, 0, 1, 0.5, 1)
    assert_match_set(results[2], 1, 2, 2, 0.333, 0.333)


def test_evaluate(dataset, mock_detector):
    result = bm.evaluate(dataset, mock_detector)
    assert_match_set(result, 4, 2, 4, 0.5, 4 / 6, 0.75)


@pytest.mark.parametrize('conf,iou,expect', [
    # High confidence threshold.
    (0.8, 0.4, (4, 2, 0, 4 / 4, 4 / 6, 0.75)),
    # Very high confidence threshold.
    (0.9, 0.4, (3, 3, 0, 3 / 3, 3 / 6, 0.75)),
    # Strict IoU, imprecise detections are not counted.
    (0.5, 0.95, (3, 3, 5, 3 / 8, 3 / 6, 0.593)),
    # Loose IoU, even very imprecise detections count.
    (0.5, 0.1, (6, 0, 2, 6 / 8, 1, 0.95)),
])
def test_evaluate_iou(dataset, mock_detector, conf, iou, expect):
    result = bm.evaluate(dataset, mock_detector, confidence_threshold=conf,
                         match_iou=iou)
    assert_match_set(result, *expect)
