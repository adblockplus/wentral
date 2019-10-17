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

"""CLI for the benchmarking functionality."""

import argparse
import json
import logging

import ady.benchmark as bm

# Logging levels set by zero, one or two -v flags.
LOGGING_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--match-iou', '-m', metavar='X', type=float, default=0.4,
        help='Minimum IOU after which the detection is considered correct '
             '(default: 0.4)',
    )
    parser.add_argument(
        '--output', '-o', metavar='JSON_FILE',
        help='Output file for the summary in JSON format',
    )
    parser.add_argument(
        '--server-url', '-s', metavar='URL',
        help='URL of ad detection service',
    )
    parser.add_argument(
        '--weights-file', '-w', metavar='PATH',
        help='Path to YOLOv3 weights file',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count', default=0,
        help='Increase the amount of debug output',
    )
    parser.add_argument(
        'dataset', metavar='DATASET',
        help='Directory that contains test images with marked ads.',
    )

    args = parser.parse_args()

    options = ['--server-url', '--weights-file']
    selected = [
        o for o in options
        if getattr(args, o[2:].replace('-', '_')) is not None
    ]

    if selected == []:
        parser.error('At least one of {} should be present'
                     .format(', '.join(options)))
    elif len(selected) > 1:
        parser.error('{} conflicts with {}'
                     .format(selected[0], ', '.join(selected[1:])))

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=LOGGING_LEVELS.get(args.verbose, logging.DEBUG))

    if args.server_url:
        import ady.client as cl
        detector = cl.ProxyAdDetector(args.server_url)
    else:  # Must have weights_file.
        import ady.detector as det
        detector = det.YoloAdDetector(args.weights_file)

    dataset = bm.LabeledDataset(args.dataset)
    evaluation = bm.evaluate(dataset, detector, args.match_iou)

    if not args.output or args.verbose > 0:
        print('Overall results:')
        print('TP:{0.tp} FN:{0.fn} FP:{0.fp}'.format(evaluation))
        print('Recall: {0.recall:.2%}'.format(evaluation))
        print('Precision: {0.precision:.2%}'.format(evaluation))

    if args.output:
        with open(args.output, 'wt', encoding='utf-8') as out_file:
            json.dump(evaluation.to_dict(), out_file, indent=2, sort_keys=True)
