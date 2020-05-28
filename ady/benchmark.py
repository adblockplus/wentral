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

"""Compare ad detections to the ground truth."""

import json
import logging
import os

import ady.utils as u
import ady.visualization as vis


def _precision(tp, fp):
    """Calculate precision from true positive and false positive counts."""
    if fp == 0:
        return 1  # by definition.
    else:
        return tp / (tp + fp)


def _recall(tp, fn):
    """Calculate recall from true positive and false negative counts."""
    if fn == 0:
        return 1  # by definition.
    else:
        return tp / (tp + fn)


def _f1(tp, fp, fn):
    """Calculate F1 score from true and false positives and false negatives."""
    p = _precision(tp, fp)
    r = _recall(tp, fn)
    if p == 0 and r == 0:
        return 0
    return 2 * p * r / (p + r)


class MatchSet:
    """Detected and expected boxes and information about their matching.

    Attributes
    ----------
    image_name : str
        Name of the image in which the ads are detected.
    detections : list of tuple (x0, y0, x1, y1, confidence, is_true)
        All detections regardless of confidence but without duplicates.
    ground_truth : list of tuple (x0, y0, x1, y1, detection_confidence)
        Expected boxes and confidence levels at which they are detected.
    confidence_threshold : float
        Minimum confidence for detections to be counted (for calculating tp,
        tn, fp, recall and precision).
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

    def __init__(self, image_name, detections, ground_truth, **params):
        """Construct the match set from detected and expected boxes.

        Parameters
        ----------
        image_name : str
            Name of the image in which the ads are detected.
        detections : list of tuple (x0, y0, x1, y1, confidence)
            Detected boxes.
        ground_truth : list of tuple (x0, y0, x1, y1)
            Expected boxes.
        params : dict
            Parameters for the match calculations, such as confidence_threshold
            and match_iou.

        """
        self.image_name = image_name
        # Sort detections by confidence (from high to low).
        self.detections = sorted(detections, key=lambda d: d[4], reverse=True)
        self.ground_truth = ground_truth
        self.confidence_threshold = params['confidence_threshold']
        self.match_iou = params['match_iou']

        self._mark_true_false()
        self._calculate_metrics()

    def _mark_true_false(self):
        """Mark detections as true or false.

        Goes through ground truth boxes and finds highest confidence matching
        detection for each of them. Ground truth record gets the confidence set
        as its detection confidece. Detection gets marked as true positive.
        Non-matched true boxes get 0 confidence and non-matched detections are
        marked as false positives.
        """
        for i in range(len(self.ground_truth)):
            true_box = self.ground_truth[i]
            for j in range(len(self.detections)):
                detection = self.detections[j]
                if len(detection) > 5:
                    continue  # Already matched to some ground truth.
                if u.iou(true_box, detection[:4]) >= self.match_iou:
                    # Add detection confidence to ground_truth box.
                    self.ground_truth[i] += (detection[4],)
                    # Mark detection as true positive.
                    self.detections[j] += (True,)
                    break
            else:  # No detections matched.
                self.ground_truth[i] += (0,)

        for j in range(len(self.detections)):
            if len(self.detections[j]) != 6:  # Wasn't matched: false positive.
                self.detections[j] += (False,)

    @property
    def detected_ground_truth(self):
        """The list of ground truth boxes that have been detected."""
        return [box for box in self.ground_truth
                if box[4] >= self.confidence_threshold]

    @property
    def missed_ground_truth(self):
        """The list of ground truth boxes that have not been detected."""
        return [box for box in self.ground_truth
                if box[4] < self.confidence_threshold]

    @property
    def true_detections(self):
        """The list of detections that matched some ground truth boxes."""
        return [det for det in self.detections
                if det[4] >= self.confidence_threshold and det[5]]

    @property
    def false_detections(self):
        """The list of detections that don't match any ground truth boxes."""
        return [det for det in self.detections
                if det[4] >= self.confidence_threshold and not det[5]]

    def _calculate_metrics(self):
        """Calculate metrics: tp, fn, fp, recall and precision."""
        self.tp = len(self.detected_ground_truth)
        self.fn = len(self.missed_ground_truth)
        self.fp = len(self.false_detections)

        self.recall = _recall(self.tp, self.fn)
        self.precision = _precision(self.tp, self.fp)
        self.f1 = _f1(self.tp, self.fp, self.fn)

        logging.info('TP:{0.tp} FN:{0.fn} FP:{0.fp} Recall:{0.recall:.2%} '
                     'Precision:{0.precision:.2%} F1:{0.f1}'.format(self))

    def to_dict(self):
        return dict(self.__dict__)


# Average precision calculation code is based on:
# https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
def _precision_recall_curve(detections, ground_truth):
    """Return precision-recall curve (PRC).

    Parameters
    ----------
    detections : list of (..., confidence, is_true)
        List of detection records as tuples with last two elements of each
        being confidence with which it's detected and the indication of whether
        it's a true positive or false positive.
    ground_truth : list of tuple
        Ground truth records. We only care about how many of them there are.

    Returns
    -------
    prc : list of (confidence, precision, recall)
        Points of precision-recall curve in the order of decreasing confidence
        (increasing recall).

    """
    tp = fp = 0
    fn = len(ground_truth)
    prc = []
    last_c = 1

    # Go through detections in decreasing confidence order.
    for detection in sorted(detections, key=lambda d: d[-2], reverse=True):
        c, is_true = detection[-2:]

        if c < last_c:
            # Detections at the same confidence level are sorted arbitrarily so
            # we only produce points when confidence changes.
            prc.append((last_c, _precision(tp, fp), _recall(tp, fn)))
            last_c = c

        if is_true:
            fn -= 1
            tp += 1
        else:
            fp += 1

    # Add last point that would not be generated inside the loop.
    prc.append((last_c, _precision(tp, fp), _recall(tp, fn)))

    # Add the end point (start point (1, 1, 0) is added by the loop). The
    # start point is needed in case the highest confidence prediction is a
    # true one, since there will be no recall=0 point in this case. The end
    # point is needed when recall=1 is not attained at any c. We take it by
    # definition that at c=0 any box is a detection so everything is recalled
    # but precision=0.
    return prc + [(0, 0, 1)]


def _interpolate_prc(prc):
    """Interpolate the precision-recall curve.

    Replaces precision with interpolated precision: the maximal precision that
    can be achieved at given or higher recall.

    Parameters
    ----------
    prc : list of (confidence, precision, recall)
        Points of precision-recall curve.

    Returns
    -------
    iprc : list of (confidence, interpolated_precision, recall)

    """
    iprc = []
    interpolated_precision = 0

    for c, precision, recall in reversed(prc):
        interpolated_precision = max(precision, interpolated_precision)
        iprc.append((c, interpolated_precision, recall))

    return list(reversed(iprc))


def _auc(prc):
    """Compute area under precision-recall curve.

    Parameters
    ----------
    prc : list of (confidence, precision, recall)
        Points of precision-recall (or interpolated precision-recall) curve.

    Returns
    -------
    auc : float
        Area under the precision-recall curve given by:

            sum((p[i+1] + p[i]) / 2 * (r[i+1] - r[i]))

    """
    return sum(
        (prc[i + 1][1] + prc[i][1]) / 2 * (prc[i + 1][2] - prc[i][2])
        for i in range(len(prc) - 1)
    )


def average_precision(detections, ground_truth):
    """Calculate average precision from detections and ground truth.

    Parameters
    ----------
    detections : list of (..., confidence, is_true)
        List of detection records as tuples with last two elements of each
        being confidence with which it's detected and the indication of whether
        it's a true positive or false positive.
    ground_truth : list of tuple
        Ground truth records. We only care about how many of them there are.

    Returns
    -------
    ap : float
        Average precision defined as area under the interpolated
        precision-recall curve.

    """
    prc = _precision_recall_curve(detections, ground_truth)
    iprc = _interpolate_prc(prc)
    return _auc(iprc)


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
    mAP : float
        Mean average precision (actually just average precision beause we have
        only one class).

    """

    def __init__(self, dataset, detector, matchsets):
        self.dataset = dataset
        self.detector = detector
        self.matchsets = matchsets
        self.image_count = len(matchsets)

        self.tp = 0
        self.fn = 0
        self.fp = 0
        for ms in matchsets:
            self.tp += ms.tp
            self.fn += ms.fn
            self.fp += ms.fp

        self.recall = _recall(self.tp, self.fn)
        self.precision = _precision(self.tp, self.fp)
        self.f1 = _f1(self.tp, self.fp, self.fn)
        self.mAP = average_precision(
            sum([ms.detections for ms in self.matchsets], []),
            sum([ms.ground_truth for ms in self.matchsets], []),
        )

    def to_dict(self):
        ret = {
            'tp': self.tp,
            'fn': self.fn,
            'fp': self.fp,
            'recall': self.recall,
            'precision': self.precision,
            'f1': self.f1,
            'mAP': self.mAP,
            'image_count': self.image_count,
            'images': [ms.to_dict() for ms in self.matchsets],
            'dataset': str(self.dataset),
            'detector': str(self.detector),
            'images_path': self.dataset.images_path
        }
        return ret

    def json_dump(self, out_file):
        """Write this evaluation into a JSON file.

        Parameters
        ----------
        out_file : file
            File to write to. It must be a text file opened in unicode mode
            with utf-8 encoding.

        """
        json.dump(self.to_dict(), out_file, indent=2, sort_keys=True)


def match_detections(dataset, detector, **params):
    """Compare regions detected by detector to the ground truth.

    Parameters
    ----------
    dataset : iterable of (image, image_path, expected_boxes)
        Source of images and true ad detections.
    detector : AdDetector (has .detect(image, path) -> list of detections)
        Ad detector to benchmark (returns a list of 5-element tuples with
        box coordinates followed by confidence).
    params : dict
        Parameters for the detector, such as confidence_threshold and
        match_iou.

    Returns
    -------
    matches : iterator of MatchSet
        Results of benchmarking.

    """
    params.setdefault('confidence_threshold', 0.5)
    params.setdefault('match_iou', 0.4)

    for image, image_path, expected_boxes in dataset:
        logging.info('Processing image: {}'.format(image_path))
        logging.debug('Marked ads: {}'.format(expected_boxes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        detected_boxes = detector.detect(image, image_path,
                                         confidence_threshold=0.001)
        logging.debug('Detected ads: {}'.format(detected_boxes))
        image_name = os.path.basename(image_path)
        ms = MatchSet(image_name, detected_boxes, expected_boxes, **params)

        if 'visualizations_path' in params:
            vis.visualize_match_set(ms, image, params['visualizations_path'])

        yield ms


def evaluate(dataset, detector, **params):
    """Evaluate the performance of detector on dataset.

    Parameters
    ----------
    dataset : iterable of (image, image_path, expected_boxes)
        Source of images and true ad detections.
    detector : AdDetector (has .detect(image, path) -> boxes)
        Ad detector to benchmark.
    params : dict
        Parameters for the detector, such as confidence_threshold and
        match_iou.

    Returns
    -------
    result : Evaluation
        Evaluation result.

    """
    if 'visualizations_path' in params:
        os.makedirs(params['visualizations_path'], exist_ok=True)
    matchsets = list(match_detections(dataset, detector, **params))
    evaluation = Evaluation(dataset, detector, matchsets)
    if 'visualizations_path' in params:
        vis.write_data_json(evaluation, params['visualizations_path'])
        vis.write_index_html(params['visualizations_path'])
    return evaluation
