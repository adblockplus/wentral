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

"""Benchmarking tool for ad detection models."""

import argparse
import logging

import wentral.benchmark as bm
import wentral.config as conf
import wentral.dataset as ds

RESULTS_TEMPLATE = """Overall results:
N: {0.image_count}
TP:{0.tp} FN:{0.fn} FP:{0.fp}
Recall: {0.recall:.2%}
Precision: {0.precision:.2%}
F1: {0.f1:.2%}
mAP: {0.mAP:.2%}"""


def parse_args():
    """Parse command line arguments."""
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

    return parser.parse_args()


def main():
    args = parse_args()
    loglevel = conf.LOGLEVELS.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=loglevel)
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
        print(RESULTS_TEMPLATE.format(evaluation))

    if args.output:
        with open(args.output, 'wt', encoding='utf-8') as out_file:
            evaluation.json_dump(out_file)
