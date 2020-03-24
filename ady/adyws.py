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

"""CLI for the ad detection web service."""

import logging
import os

import paste.translogger as tl
import waitress

import ady.yolo.detector as det
import ady.slicing_detector_proxy as sdp
import ady.webservice as ws


def make_app():
    """Prepare the ad detection web service."""
    weights_file = os.environ['YOLOv3_WEIGHTS_PATH']
    detector = det.YoloAdDetector(weights_file)
    slicing_proxy = sdp.SlicingDetectorProxy(detector)
    app = ws.make_app(slicing_proxy)
    return tl.TransLogger(app, setup_console_handler=False)


def main():
    """Expose the ad detection service using Waitress."""
    logging.getLogger().setLevel(logging.DEBUG)
    waitress.serve(make_app(), listen='*:8080')
