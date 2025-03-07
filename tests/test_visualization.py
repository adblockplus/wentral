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

"""Tests for visualization."""

import json

import pytest
from PIL import Image

import wentral.benchmark as bm
import wentral.visualization as vis


@pytest.fixture()
def match_set():
    return bm.MatchSet(
        'foo.png',
        [(10.1, 10.1, 20.1, 30.1, 0.55), (10, 40, 60, 70, 0.45)],
        [(9, 11, 22, 29), (20, 30, 70, 40), (80, 15, 95, 50)],
        confidence_threshold=0.4,
        match_iou=0.4,
    )


@pytest.fixture()
def evaluation(match_set):
    print(match_set.detected_ground_truth)
    print(match_set.missed_ground_truth)
    print(match_set.true_detections)
    print(match_set.false_detections)
    return bm.Evaluation(None, None, [match_set])


def test_vis_match_set(match_set, tmpdir):
    MARK_COLOR = (69, 69, 69)

    image = Image.new('RGB', (100, 100), (255, 255, 255))
    for box in match_set.detections + match_set.ground_truth:
        # Mark each box with a colored pixel in order to check them after they
        # are extracted.
        image.putpixel((int(box[0]) + 1, int(box[1]) + 1), MARK_COLOR)

    vis.visualize_match_set(match_set, image, str(tmpdir))
    result = Image.open(str(tmpdir.join(match_set.image_name)))

    # Check the boxes on the main image.
    assert result.getpixel((60, 70)) == (255, 255, 255)
    assert result.getpixel((10, 10)) == vis.TD_COLOR
    assert result.getpixel((19, 29)) == vis.TD_COLOR
    assert result.getpixel((10, 40)) == vis.FD_COLOR
    assert result.getpixel((9, 11)) == vis.DGT_COLOR
    assert result.getpixel((10, 11)) == vis.TD_COLOR  # TD is on top of DGT
    assert result.getpixel((20, 30)) == vis.MGT_COLOR
    assert result.getpixel((69, 39)) == vis.MGT_COLOR

    # Check the extracted boxes via marked pixels.
    for img_name in [
        'foo_fd_10,40-60,70.png',
        'foo_td_10,10-20,30.png',
        'foo_dgt_9,11-22,29.png',
        'foo_mgt_20,30-70,40.png',
        'foo_mgt_80,15-95,50.png',
    ]:
        extracted = Image.open(str(tmpdir.join(img_name)))
        assert extracted.getpixel((1, 1)) == MARK_COLOR


def test_data_js(evaluation, tmpdir):
    vis.write_data_json(evaluation, str(tmpdir))

    data = json.loads(tmpdir.join('data.json').read())
    assert data[0] == {
        'name': 'foo.png',
        'tp': 1,
        'fn': 2,
        'fp': 1,
        'detections': {
            'false': [{
                'file': 'foo_fd_10,40-60,70.png',
                'box': [10, 40, 60, 70, 0.45],
            }],
            'true': [{
                'file': 'foo_td_10,10-20,30.png',
                'box': [10.1, 10.1, 20.1, 30.1, 0.55],
            }],
        },
        'ground_truth': {
            'detected': [{
                'file': 'foo_dgt_9,11-22,29.png',
                'box': [9, 11, 22, 29, 0.55],
            }],
            'missed': [{
                'file': 'foo_mgt_20,30-70,40.png',
                'box': [20, 30, 70, 40, 0],
            }, {
                'file': 'foo_mgt_80,15-95,50.png',
                'box': [80, 15, 95, 50, 0],
            }],
        },
    }


def test_index_html(tmpdir):
    vis.write_index_html(str(tmpdir))
    index_html = tmpdir.join('index.html')
    visualization_js = tmpdir.join('visualization.js')
    assert index_html.check(exists=True)
    assert index_html.read().startswith('<html')
    assert visualization_js.check(exists=True)
    assert visualization_js.read().startswith('// Copyright')
