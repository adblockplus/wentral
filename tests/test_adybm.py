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

"""Test for the benchmarking CLI."""

import json

import pytest


def test_no_options(script_runner, dataset_dir):
    """Without --server-url or --weights-file we get an error."""
    result = script_runner.run('adybm', str(dataset_dir))
    assert not result.success
    assert 'usage:' in result.stderr
    assert 'At least one of --server-url, --weights-file' in result.stderr


def test_many_options(script_runner, dataset_dir):
    """Both --server-url and --weights-file cause an error."""
    result = script_runner.run('adybm', '-s', 'foo', '-w', 'bar',
                               str(dataset_dir))
    assert not result.success
    assert 'usage:' in result.stderr
    assert '--server-url conflicts with --weights-file' in result.stderr


@pytest.mark.script_launch_mode('inprocess')
def test_server_url(script_runner, dataset_dir, webservice):
    """Test normal operation with --server-url."""
    result = script_runner.run('adybm', '-s', webservice['url'],
                               str(dataset_dir))
    assert result.success
    assert result.stdout == """Overall results:
TP:4 FN:2 FP:4
Recall: 66.67%
Precision: 50.00%
"""
    assert result.stderr == ''


@pytest.mark.script_launch_mode('inprocess')
def test_weights_file(script_runner, mocker, dataset_dir, mock_detector):
    """Test normal operation with --weights-file."""

    def construct_detector(weights_file):
        assert weights_file == '/a/b/c'
        return mock_detector

    mocker.patch('ady.detector.YoloAdDetector', construct_detector)
    result = script_runner.run('adybm', '-w', '/a/b/c', str(dataset_dir))
    assert result.success
    assert result.stdout == """Overall results:
TP:4 FN:2 FP:4
Recall: 66.67%
Precision: 50.00%
"""
    assert result.stderr == ''


@pytest.mark.script_launch_mode('inprocess')
def test_json(script_runner, dataset_dir, tmpdir, webservice):
    """Test verbosity levels."""
    json_path = tmpdir.join('output.json')
    result = script_runner.run(
        'adybm',
        '-o', str(json_path),
        '-s', webservice['url'],
        str(dataset_dir),
    )
    assert result.success
    assert result.stdout == ''
    assert result.stderr == ''
    result = json.load(json_path.open())
    assert result['tp'] == result['fp'] == 4
    assert result['fn'] == 2


@pytest.mark.script_launch_mode('inprocess')
def test_iou(script_runner, dataset_dir, webservice):
    """Test changing the IoU."""
    result = script_runner.run('adybm', '-s', webservice['url'], '-m', '0.1',
                               str(dataset_dir))
    assert result.success
    assert result.stdout == """Overall results:
TP:6 FN:0 FP:2
Recall: 100.00%
Precision: 75.00%
"""
    assert result.stderr == ''
