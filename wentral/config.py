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

"""Common configuration and detector loading."""

import importlib

import wentral.utils as utils

DETECTOR_SHORTCUTS = {
    'json': 'wentral.json_detector.JsonDetector',
    'server': 'wentral.client.ProxyAdDetector',
    'static': 'wentral.static_detector.StaticDetector',
}


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
