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

"""Tests for the ad detector web service CLI."""

import pytest


@pytest.mark.script_launch_mode('inprocess')
@pytest.mark.parametrize('weights_file', [None, '/a/b/c'])
@pytest.mark.parametrize('via_env', [True, False])
@pytest.mark.parametrize('extras', [
    [],
    ['--slicing-threshold', '0.5', '--iou-threshold', '0.2', '--port', '80'],
])
def test_other(script_runner, mocker, shmetector, dataset_dir, weights_file,
               via_env, extras):
    """Test with -d wentral.Shmetector (that requires weights_file)."""
    cmd = [
        'wws',
        '-d', shmetector,
    ] + extras
    kwargs = {}

    if weights_file is not None:
        if via_env:
            kwargs = {'env': {'YOLOv3_WEIGHTS_PATH': weights_file}}
        else:
            cmd[3:3] = ['-w', weights_file]

    def mock_serve(app, port=None):
        """Mock for waitress.serve() that prints out the arguments."""
        app = app.application  # Unwrap from TransLogger.
        print(app.detector)
        print('port={}'.format(port))

    mocker.patch('waitress.serve', mock_serve)

    result = script_runner.run(*cmd, **kwargs)
    if weights_file is not None:
        assert result.success

        # Our mock of `waitress.server` prints the objects and their settings
        # to stdout. Check that it looks right.
        args = {
            opt: value
            for opt, value in zip(extras, extras[1:])
            if opt.startswith('-')
        }
        slicing_threshold = args.get('--slicing-threshold', 0.7)
        iou_threshold = args.get('--iou-threshold', 0.4)
        port = args.get('--port', 8080)

        assert result.stdout == (
            'SlicingDetectorProxy(detector=MD(weights_file={}, '
            'iou_threshold={}), iou_threshold={}, slice_overlap=0.2, '
            'slicing_threshold={})\nport={}\n'
        ).format(weights_file, iou_threshold, iou_threshold,
                 slicing_threshold, port)
        assert result.stderr == ''
    else:
        # There's no default for --weights-file provided by the options parser
        # and no default coming from the constructor so this should fail.
        assert not result.success
        err = 'weights_file is required for detector wentral.Shmetector'
        assert err in result.stderr
