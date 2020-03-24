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

"""CLI for the benchmarking functionality."""

import argparse
import json
import logging

import ady.benchmark as bm
import ady.config as conf

# Logging levels set by zero, one or two -v flags.
LOGGING_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

RESULTS_TEMPLATE = """Overall results:
TP:{0.tp} FN:{0.fn} FP:{0.fp}
Recall: {0.recall:.2%}
Precision: {0.precision:.2%}
mAP: {0.mAP:.2%}"""


def parse_args():
    """Parse command line arguments (and configure the logging)."""
    parser = argparse.ArgumentParser(description=__doc__)
    conf.add_detector_args(parser)

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
        '--visualizations-path', '-z', metavar='PATH',
        help='Save detection vizualizations in this directory',
    )
    parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help='Increase the amount of debug output',
    )
    parser.add_argument(
        'dataset', metavar='DATASET',
        help='Directory that contains test images with marked ads.',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=LOGGING_LEVELS.get(args.verbose, logging.DEBUG))
    detector = conf.make_detector(args)

    params = {
        'confidence_threshold': args.confidence_threshold,
        'match_iou': args.match_iou,
    }

    if args.visualizations_path:
        params['visualizations_path'] = args.visualizations_path

    dataset = bm.LabeledDataset(args.dataset)
    evaluation = bm.evaluate(dataset, detector, **params)

    if not args.output or args.verbose > 0:
        print(RESULTS_TEMPLATE.format(evaluation))

    if args.output:
        with open(args.output, 'wt', encoding='utf-8') as out_file:
            json.dump(evaluation.to_dict(), out_file, indent=2, sort_keys=True)
