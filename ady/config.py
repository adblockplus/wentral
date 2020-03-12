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

"""Common configuration and detector loading."""

import importlib
import logging

import ady.utils as utils

DETECTOR_SHORTCUTS = {
    'yolo': 'ady.yolo.detector.YoloAdDetector',
    'server': 'ady.client.ProxyAdDetector',
    'static': 'ady.static_detector.StaticDetector',
}

DEFAULT_DETECTOR = 'yolo'

# Logging levels set by zero, one or two -v flags.
LOGLEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def add_detector_args(parser):
    """Parse command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser.

    """
    parser.add_argument(
        '--detector', '-d', metavar='CLASS', default=DEFAULT_DETECTOR,
        help='Detector class (full name or shortcut: yolo, server, static; '
             'default: yolo)',
    )
    parser.add_argument(
        '--confidence-threshold', '-c', metavar='X', type=float, default=0.5,
        help='Minimum confidence for detections to be counted (default: 0.5)',
    )
    parser.add_argument(
        '--iou-threshold', metavar='X', type=float, default=0.4,
        help='IOU threshold for detection deduplication (default: 0.4)',
    )
    parser.add_argument(
        '--server-url', '-s', metavar='URL',
        help='URL of ad detection service (use with -d server)',
    )
    parser.add_argument(
        '--weights-file', '-w', metavar='PATH',
        help='Path to model weights file (use with -d yolo and other neural '
             'networks)',
    )
    parser.add_argument(
        '--path', '-p', metavar='PATH',
        help='Path to a directory with marked regions (use with -d static)',
    )


def load_detector_class(path):
    """Load detector class by path or shortcut."""
    path = DETECTOR_SHORTCUTS.get(path, path)
    try:
        module_name, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError):
        raise ImportError('Import of detector failed: ' + path)


def make_detector(args):
    """Load and instantiate the detector class.

    Parameters
    ----------
    args : Namespace
        Namespace containing parsed arguments (see `add_detector_ads`).

    Returns
    -------
    detector : AdDetector
        Initialized ad detector object.

    """
    detector_class = load_detector_class(args.detector)
    kw = utils.kwargs_from_ns(detector_class, args)
    return detector_class(**kw)
