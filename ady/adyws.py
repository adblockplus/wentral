# This file is part of Ad Detect YOLO <https://adblockplus.org/>,
# Copyright (C) 2019-present eyeo GmbH
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

"""Ad detection web service."""

import argparse
import logging
import os

import paste.translogger as tl
import waitress

import ady.config as conf
import ady.slicing_detector_proxy as sdp
import ady.utils as utils
import ady.webservice as ws


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
