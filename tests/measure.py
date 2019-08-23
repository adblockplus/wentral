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

"""Measure performance of ad detection web services."""

import argparse
import json
import logging
import os

import ady.client as client
import bimed.index as idx

# Logging levels set by zero, one or two -v flags.
LOGGING_LEVELS = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}

# At this or greater IOU two boxes are considered to be a match.
MATCH_IOU = 0.4


def load_index(imgdir, summary):
    """Load the index and return an iterator of (image, ad_regions)."""
    index = idx.reg_index(imgdir)
    ad_region_types = [rt for rt in index.region_types if 'label' not in rt]
    summary['flags']['ad_region_types'] = ad_region_types
    logging.info('Ad region types: {}'.format(ad_region_types))
    for image in sorted(index):
        ad_regions = [
            region[:4] for region in index[image]
            if region[4] in ad_region_types
        ]
        yield image, ad_regions


def iou(box1, box2):
    """Calculate intersection over union of two boxes."""

    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    if int_x0 > int_x1 or int_y0 > int_y1:  # No intersection.
        return 0

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    return int_area / (b1_area + b2_area - int_area + 1e-05)


def compare(detected, expected, match_iou=0.4):
    """Compare detected boxes to expected boxes.

    Returns a tuple: (true_positives, false_negatives, false_positives) where:
    - true_positives is the number of expected regions that were detected,
    - false_negatives is the number of expected regions that were not detected,
    - false_positives is the number of detected regions that were not expected.

    """
    matches = {}
    for eb in expected:
        for db in detected:
            if iou(eb, db) >= MATCH_IOU:
                matches[eb] = db
    tp = len(matches)
    fn = len(expected) - tp
    fp = len(detected) - tp
    recall = tp / (tp + fn + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    logging.info('TP:{} FN:{} FP:{} Recall:{:.2%} Precision:{:.2%}'
                 .format(tp, fn, fp, recall, precision))
    return tp, fn, fp


def measure(url, imgdir, match_iou):
    """Run the measurements."""
    logging.debug('Using images from {} and server at {}'.format(imgdir, url))
    summary = {
        'flags': {
            'input_dir': imgdir,
            'match_iou': match_iou,
            'server_url': url,
        },
        'images': [],
    }

    for image, ad_regions in load_index(imgdir, summary):
        logging.info('Processing image: {}'.format(image))
        logging.debug('Marked ad regions: {}'.format(ad_regions))

        with open(os.path.join(imgdir, image), 'rb') as image_file:
            detected_ads = client.detect_ads(image_file, url)
        detected_ads = [d[:4] for d in detected_ads]
        logging.debug('Detected ads: {}'.format(detected_ads))

        tp, fn, fp = compare(detected_ads, ad_regions, match_iou)
        summary['images'].append({
            'image': image,
            'detected_boxes': detected_ads,
            'marked_boxes': ad_regions,
            'tp': tp,
            'fn': fn,
            'fp': fp,
        })

    return summary


def finalize_summary(summary):
    """Compute totals from the measurement summary and display them."""
    stats = {
        s: sum(i[s] for i in summary['images'])
        for s in ['tp', 'fn', 'fp']
    }
    stats['recall'] = stats['tp'] / (stats['tp'] + stats['fn'])
    stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp'])
    summary['stats'] = stats
    print('Overall results:')
    print('TP:{0[tp]} FN:{0[fn]} FP:{0[fp]}'.format(stats))
    print('Recall: {0[recall]:.2%}'.format(stats))
    print('Precision: {0[precision]:.2%}'.format(stats))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'imgdir', metavar='DIR',
        help='Directory that contains test images with marked ads.',
    )
    parser.add_argument(
        '--match-iou', '-m', metavar='X',
        type=float, default=0.4,
        help='Minimum IOU after which the detection is considered correct '
             '(default: 0.4)',
    )
    parser.add_argument(
        '--output', '-o', metavar='JSON_FILE',
        help='Output file for the summary in JSON format',
    )
    parser.add_argument(
        '--server-url', '-s', metavar='URL',
        default='http://localhost:8080/',
        help='URL of ad detection service (default: http://localhost:8080/)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count', default=0,
        help='Increase the amount of debug output',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=LOGGING_LEVELS.get(args.verbose, logging.DEBUG))
    summary = measure(args.server_url, args.imgdir, args.match_iou)
    finalize_summary(summary)
    if args.output:
        with open(args.output, 'wt', encoding='utf-8') as out_file:
            json.dump(summary, out_file, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
