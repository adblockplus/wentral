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


@pytest.mark.script_launch_mode('inprocess')
def test_server(script_runner, dataset_dir, webservice):
    """Test with -d server and --server-url."""
    result = script_runner.run(
        'adybm',
        '-d', 'server',
        '-s', webservice['url'],
        str(dataset_dir),
    )
    assert result.success
    assert result.stdout == """Overall results:
N: 3
TP:4 FN:2 FP:4
Recall: 66.67%
Precision: 50.00%
F1: 57.14%
mAP: 75.00%
"""
    assert result.stderr == ''


@pytest.mark.script_launch_mode('inprocess')
@pytest.mark.parametrize('weights_file', [None, '/a/b/c'])
def test_other(script_runner, shmetector, dataset_dir, weights_file):
    """Test with -d ady.Shmetector (that requires weights_file)."""
    cmd = [
        'adybm',
        '-d', shmetector,
        str(dataset_dir),
    ]
    if weights_file is not None:
        cmd[3:3] = ['-w', weights_file]

    result = script_runner.run(*cmd)
    if weights_file is not None:
        assert result.success
        assert result.stdout == """Overall results:
N: 3
TP:4 FN:2 FP:4
Recall: 66.67%
Precision: 50.00%
F1: 57.14%
mAP: 75.00%
"""
        assert result.stderr == ''
    else:
        # There's no default for --weights-file provided by the options parser
        # and no default coming from the constructor so this should fail.
        assert not result.success
        err = 'Parameter weights_file is required for detector ady.Shmetector'
        assert err in result.stderr


def test_static(script_runner, dataset_dir):
    """Test with -d static and --path."""
    result = script_runner.run(
        'adybm',
        '-d', 'static',
        '-p', str(dataset_dir),
        str(dataset_dir),
    )
    assert result.success
    assert result.stdout == """Overall results:
N: 3
TP:6 FN:0 FP:0
Recall: 100.00%
Precision: 100.00%
F1: 100.00%
mAP: 100.00%
"""
    assert result.stderr == ''


@pytest.fixture()
def dataset_copy(dataset_dir, tmpdir):
    """Copy of the dataset (usually so we can modify it)."""
    dataset_copy = tmpdir.join('dataset_copy')
    dataset_dir.copy(dataset_copy)
    return dataset_copy


def test_static_missing(script_runner, dataset_dir, dataset_copy):
    """Test static with some images missing."""
    dataset_copy.join('1.png').remove()
    dataset_copy.join('index.csv').write('')

    result = script_runner.run(
        'adybm',
        '-d', 'static',
        '-p', str(dataset_copy),
        str(dataset_dir),
    )
    assert not result.success
    assert 'Regions information is missing for 1.png' in result.stderr


@pytest.mark.script_launch_mode('inprocess')
def test_json(script_runner, dataset_dir, tmpdir, webservice):
    """Test verbosity levels."""
    json_path = tmpdir.join('output.json')
    result = script_runner.run(
        'adybm',
        '-d', 'server',
        '-o', str(json_path),
        '-s', webservice['url'],
        str(dataset_dir),
    )
    assert result.success
    assert result.stdout == ''
    assert result.stderr == ''
    result = json.load(json_path.open())
    assert result['image_count'] == 3
    assert result['tp'] == result['fp'] == 4
    assert result['fn'] == 2
    assert result['precision'] == pytest.approx(0.5, 0.001)
    assert result['recall'] == pytest.approx(0.6667, 0.001)
    assert result['mAP'] == pytest.approx(0.75, 0.001)
    assert result['f1'] == pytest.approx(0.5714, 0.001)


@pytest.mark.script_launch_mode('inprocess')
def test_iou(script_runner, dataset_dir, webservice):
    """Test changing the IoU."""
    result = script_runner.run(
        'adybm',
        '-d', 'server',
        '-s', webservice['url'],
        '-m', '0.1',
        str(dataset_dir),
    )
    assert result.success
    assert result.stdout == """Overall results:
N: 3
TP:6 FN:0 FP:2
Recall: 100.00%
Precision: 75.00%
F1: 85.71%
mAP: 95.14%
"""
    assert result.stderr == ''


@pytest.mark.script_launch_mode('inprocess')
def test_visualize_out_files(script_runner, dataset_dir, tmpdir, webservice):
    """Test vizualizing the detection boxes."""
    vis_images = {
        i.basename for i in dataset_dir.listdir()
        if i.ext == '.png'
    }
    vis_ui = {'data.js', 'index.html', 'visualization.js'}
    vis_dir = tmpdir.join('vis_dir')

    result = script_runner.run(
        'adybm',
        '-d', 'server',
        '-s', webservice['url'],
        '-z', str(vis_dir),
        str(dataset_dir),
    )

    assert result.success
    assert vis_dir.check(dir=1)
    assert {i.basename for i in vis_dir.listdir()} == vis_images | vis_ui

    # We don't check that the boxes are drawn correctly and that the data in
    # data.js is right. The tests in test_visualization.py check that and we
    # trust that this is enough.
