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

import json

import pytest
from PIL import Image

import ady.benchmark as bm
import ady.visualization as vis


@pytest.fixture()
def match_set():
    return bm.MatchSet(
        'foo.png',
        [(10, 10, 20, 30, 0.55), (10, 40, 60, 70, 0.45)],
        [(9, 11, 22, 29), (20, 30, 70, 40), (80, 15, 95, 50)],
        confidence_threshold=0.4,
        match_iou=0.4,
    )


def test_vis_match_set(match_set, tmpdir):
    print(match_set.detected_ground_truth)
    print(match_set.missed_ground_truth)
    print(match_set.true_detections)
    print(match_set.false_detections)

    image = Image.new('RGB', (100, 100), (255, 255, 255))
    result = vis.visualize_match_set(match_set, image)

    assert result.getpixel((60, 70)) == (255, 255, 255)
    assert result.getpixel((10, 10)) == vis.TD_COLOR
    assert result.getpixel((19, 29)) == vis.TD_COLOR
    assert result.getpixel((10, 40)) == vis.FD_COLOR
    assert result.getpixel((9, 11)) == vis.DGT_COLOR
    assert result.getpixel((10, 11)) == vis.TD_COLOR  # TD is on top of DGT
    assert result.getpixel((20, 30)) == vis.MGT_COLOR
    assert result.getpixel((69, 39)) == vis.MGT_COLOR


def test_data_js(match_set, tmpdir):
    evaluation = bm.Evaluation(None, None, [match_set])
    vis.write_data_js(evaluation, str(tmpdir))

    data_js_content = tmpdir.join('data.js').read()
    assert data_js_content.startswith('imageData = [')
    assert data_js_content.endswith('];')
    data = json.loads(data_js_content[12:-1])
    assert data[0] == {'name': 'foo.png', 'tp': 1, 'fn': 2, 'fp': 1}


def test_index_html(tmpdir):
    vis.write_index_html(str(tmpdir))
    index_html = tmpdir.join('index.html')
    visualization_js = tmpdir.join('visualization.js')
    assert index_html.check(exists=True)
    assert index_html.read().startswith('<html')
    assert visualization_js.check(exists=True)
    assert visualization_js.read().startswith('// This')
