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

"""Configuration and detector loading."""

import importlib
import inspect

DETECTOR_SHORTCUTS = {
    'json': 'wentral.json_detector.JsonDetector',
    'server': 'wentral.client.ProxyDetector',
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


def kwargs_from_ns(func, args):
    """Extract kwargs for calling `func` from argparse namespace."""
    params = inspect.signature(func).parameters
    kwargs = {}
    for k, v in params.items():
        if getattr(args, k, None) is not None:
            kwargs[k] = getattr(args, k)
        elif v.kind in {inspect.Parameter.VAR_KEYWORD,
                        inspect.Parameter.VAR_POSITIONAL}:
            # Skip varargs (*args, **kwargs).
            continue
        elif v.default == v.empty:
            t = str(func).split("'")
            func_name = t[1] if len(t) == 3 else func
            raise Exception('Parameter {} is required for detector {}'
                            .format(k, func_name))
    for extra in getattr(args, 'extra', []):
        try:
            name, value = extra.split('=')
            if name in params:
                annotation = params[name].annotation
                if annotation != inspect.Parameter.empty:
                    value = annotation(value)
        except ValueError:
            raise Exception('Invalid format of extra argument: ' + extra)
        kwargs[name] = value
    return kwargs


def make_detector(args):
    """Load and instantiate the detector class.

    Parameters
    ----------
    args : Namespace
        Namespace containing parsed arguments (see `kwargs_from_ns`).

    Returns
    -------
    detector : Detector
        Initialized object detector instance.

    """
    detector_class = load_detector_class(args.detector)
    kw = kwargs_from_ns(detector_class, args)
    return detector_class(**kw)
