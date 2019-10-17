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

"""Compare ad detections to the ground truth."""

import logging
import os

import bimed.index as idx
import PIL

# Add this to a possibly zero-valued denominator to avoid division by zero.
EPSILON = 1e-7


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

    return int_area / (b1_area + b2_area - int_area + EPSILON)


class MatchSet:
    """Detected and expected boxes and information about their matching.

    Attributes
    ----------
    image_name : str
        Name of the image in which the ads are detected.
    detected : list of tuple
        Detected boxes.
    expected : list of tuple
        Expected boxes (a.k.a. ground truth).
    match_iou : float
        IoU cutoff used for matching detected with expected.
    tp : int
        Number of true positives (expected and detected).
    fn : int
        Number of false negatives (expected but not detected).
    fp : int
        Number of false positives (not expected but detected).
    recall : float
        Ratio of expected boxes that was detected.
    precision : float
        Ratio of detected boxes that was expected.

    """

    def __init__(self, image_name, detected, expected, match_iou):
        """Construct the match set from detected and expected boxes.

        Parameters
        ----------
        image_name : str
            Name of the image in which the ads are detected.
        detected : list of tuple
            Detected boxes (tuples of 4 elements)
        expected : list of tuple
            Expected boxes (tuples of 4 elements)
        match_iou : float
            Minimum IoU (intersection over union) where the boxes are
            considered to be matching.

        """
        self.image_name = image_name
        self.detected = detected
        self.expected = expected
        self.match_iou = match_iou

        # Several detections might match the same expected region. They are
        # counted as one match. It's not possible for one detection to match
        # multiple expected regions (by design).
        matches = {}
        for eb in expected:
            for db in detected:
                if iou(eb, db) >= match_iou:
                    matches[eb] = db
                    break

        self.tp = len(matches)
        self.fn = len(expected) - self.tp
        self.fp = len(detected) - self.tp

        self.recall = self.tp / (self.tp + self.fn + EPSILON)
        self.precision = self.tp / (self.tp + self.fp + EPSILON)

        logging.info('TP:{0.tp} FN:{0.fn} FP:{0.fp} Recall:{0.recall:.2%} '
                     'Precision:{0.precision:.2%}'.format(self))

    def to_dict(self):
        return dict(self.__dict__)


class Evaluation:
    """Summary of detections, their accuracy and overall statistics.

    Attributes
    ----------
    dataset : LabeledDataset
        Source of images and ground truth.
    detector : AdDetector (has .detect(image, path) -> boxes)
        Ad detector that was evaluated.
    matchsets : list of MatchSet
        Information about performance on individual images.
    tp : int
        Number of true positives (expected and detected).
    fn : int
        Number of false negatives (expected but not detected).
    fp : int
        Number of false positives (not expected but detected).
    recall : float
        Ratio of expected boxes that was detected.
    precision : float
        Ratio of detected boxes that was expected.

    """

    def __init__(self, dataset, detector, matchsets):
        self.dataset = dataset
        self.detector = detector
        self.matchsets = matchsets

        self.tp = 0
        self.fn = 0
        self.fp = 0
        for ms in matchsets:
            self.tp += ms.tp
            self.fn += ms.fn
            self.fp += ms.fp

        self.recall = self.tp / (self.tp + self.fn + EPSILON)
        self.precision = self.tp / (self.tp + self.fp + EPSILON)

    def to_dict(self):
        ret = {
            'tp': self.tp,
            'fn': self.fn,
            'fp': self.fp,
            'recall': self.recall,
            'precision': self.precision,
            'images': [ms.to_dict() for ms in self.matchsets],
            'dataset': str(self.dataset),
            'detector': str(self.detector),
        }
        return ret


class LabeledDataset:
    """A set of images with marked regions.

    Attributes
    ----------
    path : str
        Path to the images and region files.
    index : FragmentIndex
        Index of regions.
    ad_region_types : list of str
        Region types that are considered ads.

    """

    def __init__(self, path):
        self.path = path
        self.index = idx.reg_index(path)
        self.ad_region_types = [
            rt for rt in self.index.region_types
            if 'label' not in rt
        ]
        logging.debug('Ad region types: {}'.format(self.ad_region_types))

    def __str__(self):
        return 'LabeledDataset({})'.format(self.path)

    def __iter__(self):
        """Yield images, paths and marked boxes.

        Yields
        ------
        image_data : (Image, set, list of tuple)
            Images, their paths and ad boxes.

        """
        for image_name in sorted(self.index):
            image_path = os.path.join(self.path, image_name)
            ad_boxes = [
                region[:4] for region in self.index[image_name]
                if region[4] in self.ad_region_types
            ]
            yield PIL.Image.open(image_path), image_path, ad_boxes

    def detect(self, image, image_path):
        """Return ad detections based on marked regions.

        Raises
        ------
        KeyError
            If there are no regions for specific image in this dataset.

        """
        image_name = os.path.basename(image_path)
        try:
            regions = self.index[image_name]
        except KeyError:
            raise Exception('Regions information is missing for {} in {}'
                            .format(image_name, self.path))

        return [
            region[:4] for region in regions
            if region[4] in self.ad_region_types
        ]


def match_detections(dataset, detector, match_iou=0.4):
    """Compare regions detected by detector to the ground truth.

    Parameters
    ----------
    dataset : iterable of (image, image_path, expected_boxes)
        Source of images and true ad detections.
    detector : AdDetector (has .detect(image, path) -> boxes)
        Ad detector to benchmark.
    match_iou : float
        Minimum IoU (intersection over union) where the boxes are considered to
        be matching.

    Returns
    -------
    matches : iterator of MatchSet
        Results of benchmarking.

    """
    for image, image_path, expected_boxes in dataset:
        logging.info('Processing image: {}'.format(image_path))
        logging.debug('Marked ads: {}'.format(expected_boxes))
        detected_boxes = [d[:4] for d in detector.detect(image, image_path)]
        logging.debug('Detected ads: {}'.format(detected_boxes))
        image_name = os.path.basename(image_path)
        yield MatchSet(image_name, detected_boxes, expected_boxes, match_iou)


def evaluate(dataset, detector, match_iou=0.4):
    """Evaluate the performance of detector on dataset.

    Parameters
    ----------
    dataset : iterable of (image, image_path, expected_boxes)
        Source of images and true ad detections.
    detector : AdDetector (has .detect(image, path) -> boxes)
        Ad detector to benchmark.
    match_iou : float
        Minimum IoU (intersection over union) where the boxes are considered to
        be matching.

    Returns
    -------
    result : Evaluation
        Evaluation result.

    """
    matchsets = list(match_detections(dataset, detector, match_iou))
    return Evaluation(dataset, detector, matchsets)
