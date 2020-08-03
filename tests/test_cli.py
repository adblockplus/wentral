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

"""Tests for the command line interface."""

import json

import pytest

# This is the benchmark output from running mock_detector on the test dataset.
MOCK_BM_OUTPUT = """Overall results:
N: 3
TP:4 FN:2 FP:4
Recall: 66.67%
Precision: 50.00%
F1: 57.14%
mAP: 75.00%
"""


@pytest.mark.script_launch_mode('inprocess')
def test_server(script_runner, dataset_dir, webservice):
    """Test with -d server and --server-url."""
    result = script_runner.run(
        'wentral', 'benchmark',
        '-d', 'server',
        '-s', webservice['url'],
        str(dataset_dir),
    )
    assert result.success
    assert result.stdout == MOCK_BM_OUTPUT
    assert result.stderr == ''


@pytest.mark.script_launch_mode('inprocess')
@pytest.mark.parametrize('weights_file', [None, '/a/b/c'])
@pytest.mark.parametrize('extras', [
    [],
    ['-x', 'broken'],
    ['--extra', 'extra_one=foo', '-x', 'extra_two=bar'],
])
def test_other(script_runner, shmetector, dataset_dir, weights_file, extras):
    """Test with -d wentral.Shmetector (that requires weights_file)."""
    cmd = [
        'wentral', 'bm',
        '-d', shmetector,
        str(dataset_dir),
    ] + extras
    if weights_file is not None:
        cmd[4:4] = ['-w', weights_file]

    result = script_runner.run(*cmd)
    if weights_file is not None:
        expected_output = MOCK_BM_OUTPUT
        if extras:
            if 'broken' in extras:
                assert not result.success
                assert 'Invalid format of extra argument' in result.stderr
                return

            for i in range(1, len(extras), 2):
                expected_output = extras[i] + '\n' + expected_output

        assert result.success
        assert result.stdout == expected_output
        assert result.stderr == ''
    else:
        # There's no default for --weights-file provided by the options parser
        # and no default coming from the constructor so this should fail.
        assert not result.success
        err = 'weights_file is required for detector wentral.Shmetector'
        assert err in result.stderr


def test_static(script_runner, dataset_dir):
    """Test with -d static and --path."""
    result = script_runner.run(
        'wentral', 'bm',
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
        'wentral', 'bm',
        '-d', 'static',
        '-p', str(dataset_copy),
        str(dataset_dir),
    )
    assert not result.success
    assert 'Regions information is missing for 1.png' in result.stderr


@pytest.mark.script_launch_mode('inprocess')
def test_json_output(script_runner, dataset_dir, tmpdir, webservice):
    """Test JSON output."""
    json_path = tmpdir.join('output.json')
    result = script_runner.run(
        'wentral', 'bm',
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
@pytest.mark.parametrize('extra_args', [[], ['-c', '0.9']])
def test_json_detector(script_runner, dataset_dir, json_output, extra_args):
    """Test -d json and loading ground truth from the same JSON file."""
    cmd = [
        'wentral', 'bm',
        '-d', 'json',
        '-p', str(json_output),
        str(dataset_dir),
    ]
    if extra_args:
        cmd[-1:-1] = extra_args

    result = script_runner.run(*cmd)

    assert result.success
    assert result.stderr == ''

    if extra_args:
        # With high confidence threshold there are no false positives.
        assert 'Precision: 100.00%' in result.stdout
    else:
        assert 'Precision: 50.00%' in result.stdout


@pytest.mark.script_launch_mode('inprocess')
def test_json_dataset(script_runner, webservice, json_output, tmpdir):
    """Test loading the dataset from a JSON file."""
    json_output2 = tmpdir.join('output2.json')
    result = script_runner.run(
        'wentral', 'bm',
        '-d', 'server',
        '-s', webservice['url'],
        '-o', str(json_output2),
        str(json_output),
    )
    assert result.success
    assert result.stdout == ''
    assert result.stderr == ''

    # Do it again with the output of the run above.
    result = script_runner.run(
        'wentral', 'bm',
        '-d', 'server',
        '-s', webservice['url'],
        str(json_output2),
    )
    assert result.success
    assert result.stdout == MOCK_BM_OUTPUT
    assert result.stderr == ''


@pytest.mark.script_launch_mode('inprocess')
def test_iou(script_runner, dataset_dir, webservice):
    """Test changing the IoU."""
    result = script_runner.run(
        'wentral', 'bm',
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
    vis_expect = {
        # Visualization images.
        '0.png',
        '0_dgt_0,0-50,20.png',
        '0_dgt_80,10-95,50.png',
        '0_fd_0,70-50,90.png',
        '0_td_0,0-50,20.png',
        '0_td_82,12-94,45.png',
        '1.png',
        '1_dgt_10,10-80,25.png',
        '1_fd_10,30-30,60.png',
        '1_td_10,10-80,25.png',
        '2.png',
        '2_dgt_60,60-90,90.png',
        '2_fd_20,20-50,50.png',
        '2_fd_5,5-15,15.png',
        '2_mgt_0,0-20,20.png',
        '2_mgt_30,30-40,40.png',
        '2_td_60,60-90,90.png',
        # Data.
        'data.json',
        # The UI itself.
        'index.html',
        'visualization.js',
    }
    vis_dir = tmpdir.join('vis_dir')

    result = script_runner.run(
        'wentral', 'bm',
        '-d', 'server',
        '-s', webservice['url'],
        '-z', str(vis_dir),
        str(dataset_dir),
    )

    assert result.success
    assert vis_dir.check(dir=1)
    assert {i.basename for i in vis_dir.listdir()} == vis_expect

    # We don't check that the boxes are drawn and extracted correctly and that
    # the data in data.js is right. The tests in test_visualization.py check
    # that and we trust that this is enough.
