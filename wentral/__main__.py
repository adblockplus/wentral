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

"""Web front end and benchmarking tool for web object detection models."""

import argparse
import logging
import sys

import paste.translogger as tl
import waitress

import wentral.benchmark as bm
import wentral.config as conf
import wentral.dataset as ds
import wentral.slicing_detector_proxy as sdp
import wentral.webservice as ws

parser = argparse.ArgumentParser(description=__doc__)
subparsers = parser.add_subparsers(help='sub-command help')


def command(*args, **kw):
    """Return a decorator for command functions.

    This decorator will create a subparser for the command function passing
    all the arguments of `command()` to `.add_parser()`. If no name and help
    are provided for the command, they will be taken from the function name
    and docstring (only the first line of the docstring is used) of the
    decorated function.

    """

    def decorator(func):
        nonlocal args, kw

        if not args:
            args = [func.__name__]
        if 'help' not in kw:
            kw['help'] = func.__doc__.split('\n')[0]

        cmd = subparsers.add_parser(*args, **kw)
        for arg in reversed(getattr(func, '__args__', [])):
            cmd.add_argument(*arg['args'], **arg['kw'])
        cmd.set_defaults(func=func)

        return func

    return decorator


def arg(*args, **kw):
    """Return a decorator that will add an argument to a command function.

    All parameters passed to the decorator will be passed to `.add_argument()`
    call of the subparser corresponding to the decorated function.

    """

    def decorator(func):
        nonlocal args, kw

        if not hasattr(func, '__args__'):
            func.__args__ = []
        func.__args__.append({'args': args, 'kw': kw})

        return func

    return decorator


def common_args():
    """Add common arguments using @arg decorator."""
    arg_decorators = [
        arg(
            '--detector', '-d', metavar='CLASS',
            help='Detector class (full name or shortcut, e.g. server, static, '
                 'json, wentral.client.ProxyDetector)',
        ),
        arg(
            '--confidence-threshold', '-c', metavar='X', type=float,
            default=0.5,
            help='Minimum confidence for detections to be counted '
                 '(default: 0.5)',
        ),
        arg(
            '--iou-threshold', metavar='X', type=float, default=0.4,
            help='IOU threshold for detection deduplication (default: 0.4)',
        ),
        arg(
            '--server-url', '-s', metavar='URL',
            help='URL of detection service (use with -d server)',
        ),
        arg(
            '--weights-file', '-w', metavar='PATH',
            help='Path to model weights file '
                 '(use with neural network detectors)',
        ),
        arg(
            '--path', '-p', metavar='PATH',
            help='Path to a directory with marked regions '
                 '(use with -d static)',
        ),
        arg(
            '--extra', '-x', metavar='ARGNAME=VALUE', action='append',
            default=[],
            help='Pass an extra argument to the detector',
        ),
    ]

    def decorator(func):
        for arg_decorator in arg_decorators:
            func = arg_decorator(func)
        return func

    return decorator


BM_RESULTS_TEMPLATE = """Overall results:
N: {0.image_count}
TP:{0.tp} FN:{0.fn} FP:{0.fp}
Recall: {0.recall:.2%}
Precision: {0.precision:.2%}
F1: {0.f1:.2%}
mAP: {0.mAP:.2%}"""


@command(aliases=['bm'])
@common_args()
@arg(
    '--match-iou', '-m', metavar='X', type=float, default=0.4,
    help='Minimum IOU after which the detection is considered correct '
         '(default: 0.4)',
)
@arg(
    '--output', '-o', metavar='JSON_FILE',
    help='Output file for the summary in JSON format',
)
@arg(
    '--visualizations-path', '-z', metavar='PATH',
    help='Save detection vizualizations in this directory',
)
@arg(
    '--verbose', '-v', action='count', default=0,
    help='Increase the amount of debug output',
)
@arg(
    'dataset', metavar='DATASET',
    help='Directory that contains test images with marked objects.',
)
def benchmark(args):
    """Measure and visualize model performance."""
    detector = conf.make_detector(args)

    params = {
        'confidence_threshold': args.confidence_threshold,
        'match_iou': args.match_iou,
    }

    if args.visualizations_path:
        params['visualizations_path'] = args.visualizations_path

    if args.dataset.endswith('.json'):
        dataset = ds.JsonDataset(args.dataset)
    else:
        dataset = ds.LabeledDataset(args.dataset)

    evaluation = bm.evaluate(dataset, detector, **params)

    if not args.output or args.verbose > 0:
        print(BM_RESULTS_TEMPLATE.format(evaluation))

    if args.output:
        with open(args.output, 'wt', encoding='utf-8') as out_file:
            evaluation.json_dump(out_file)


@command(aliases=['ws'])
@common_args()
@arg(
    '--verbose', '-v', action='count', default=0,
    help='Increase the amount of debug output',
)
@arg(
    '--port', type=int, default=8080, metavar='N',
    help='Port to listen on (default: 8080)',
)
@arg(
    '--slicing-threshold', type=float, default=0.7, metavar='X',
    help='Aspect ratio beyond which the input image will be divided into '
         'square slices (default: 0.7)',
)
@arg(
    '--slice-overlap', type=float, default=0.2, metavar='X',
    help='Overlap ratio for slices of non-square images (default: 0.2)',
)
def webserve(args):
    """Make the detector available as an HTTP web service."""
    detector = conf.make_detector(args)
    # Remove/change arguments that are only relevant for the main detector.
    args.detector = detector
    args.extra = []
    kw = conf.kwargs_from_ns(sdp.SlicingDetectorProxy, args)
    slicing_proxy = sdp.SlicingDetectorProxy(**kw)
    app = ws.make_app(slicing_proxy)
    lapp = tl.TransLogger(app, setup_console_handler=False)
    waitress.serve(lapp, port=args.port)


# Logging levels set by zero, one or two -v flags.
LOGLEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def configure_logging(args):
    """Configure logging based on verbose option."""
    loglevel = LOGLEVELS.get(args.verbose, logging.INFO)
    logging.basicConfig(level=loglevel)


def main():
    """Run the CLI."""
    args = parser.parse_args()
    if callable(getattr(args, 'func', None)):
        configure_logging(args)
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)
