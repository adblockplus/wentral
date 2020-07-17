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

"""Ad detection web service."""

import argparse
import logging
import os

import paste.translogger as tl
import waitress

import wentral.config as conf
import wentral.slicing_detector_proxy as sdp
import wentral.utils as utils
import wentral.webservice as ws


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    conf.add_detector_args(parser)

    parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help='Increase the amount of debug output',
    )
    parser.add_argument(
        '--port', type=int, default=8080, metavar='N',
        help='Port to listen on (default: 8080)',
    )
    parser.add_argument(
        '--slicing-threshold', type=float, default=0.7, metavar='X',
        help='Aspect ratio beyond which the input image will be divided into '
             'square slices (default: 0.7)',
    )
    parser.add_argument(
        '--slice-overlap', type=float, default=0.2, metavar='X',
        help='Overlap ratio for slices of non-square images (default: 0.2)',
    )

    # For backward compatibility we support specifying the weights file via an
    # environment variable.
    default_weights_file = os.environ.get('YOLOv3_WEIGHTS_PATH', None)
    parser.set_defaults(weights_file=default_weights_file)

    return parser.parse_args()


def make_app(detector):
    """Prepare the ad detection web service."""
    app = ws.make_app(detector)
    return tl.TransLogger(app, setup_console_handler=False)


def main():
    """Expose the ad detection service using Waitress."""
    args = parse_args()
    loglevel = conf.LOGLEVELS.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=loglevel)
    detector = conf.make_detector(args)
    kw = utils.kwargs_from_ns(sdp.SlicingDetectorProxy, args)
    del kw['detector']
    slicing_proxy = sdp.SlicingDetectorProxy(detector, **kw)
    app = make_app(slicing_proxy)
    waitress.serve(app, port=args.port)
