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

"""Slicing detector proxy.

Cuts full page screenshots into square slices, runs detections on each slice
and then combines them.
"""

import functools
import math

import wentral.ad_detector as ad
import wentral.constants as const
import wentral.utils as utils


def _to_absolute(detection, box):
    """Convert detection relative to box to absolute coordinates."""
    x0, y0, x1, y1, p = detection
    bx0, by0, *_ = box
    return x0 + bx0, y0 + by0, x1 + bx0, y1 + by0, p


class SlicingDetectorProxy(ad.AdDetector):
    """Detects ads in full page screenshots (that are tall).

    If the input image aspect ratio (short side over long size) is less than
    slicing_threshold, it is cut into slices (that overlap by slice_overlap)
    and each of them is passed to the wrapped detector. The detections from all
    slices are combined and deduplicated.

    """

    def __init__(self, detector, iou_threshold=const.IOU_THRESHOLD,
                 slicing_threshold=const.SLICING_THRESHOLD,
                 slice_overlap=const.SLICE_OVERLAP):
        """Constructor.

        Parameters
        ----------
        detector : Detector
            A class with `detect(image, path, ...)` method that is used to
            detect ads in square slices.
        iou_threshold : float
            Determines which detections from adjacent slices are considered the
            same.
        slicing_threshold : float
            Aspect ratio threshold after which input images will be sliced.
        slice_overlap : float
            Percentage of overlap between adjacent slices.

        """
        super().__init__(
            detector=detector,
            iou_threshold=iou_threshold,
            slicing_threshold=slicing_threshold,
            slice_overlap=slice_overlap,
        )

    @classmethod
    def _slice_boxes(cls, image_size, slicing_threshold, slice_overlap):
        """Calculate slice positions (to pass to image.crop())."""
        x_size, y_size = image_size

        if x_size * slicing_threshold > y_size:
            # Width >> height -- swap x and y, slice, unswap x and y.
            xy_swap_slices = cls._slice_boxes(
                (y_size, x_size),
                slicing_threshold,
                slice_overlap,
            )
            return list(map(utils.xy_swap, xy_swap_slices))

        if y_size * slicing_threshold <= x_size:
            # Close enough to square -- don't slice.
            return [(0, 0, x_size, y_size)]

        # Height >> width -- do vertical slicing.
        overlap_pixels = int(x_size * slice_overlap)
        slice_count = int(math.ceil(
            (y_size - overlap_pixels) / (x_size - overlap_pixels)
        ))
        slice_step = (y_size - x_size) / (slice_count - 1)
        slice_starts = [int(slice_step * i) for i in range(slice_count)]
        slice_starts[-1] = y_size - x_size  # Correct for rounding.

        return [(0, start, x_size, start + x_size) for start in slice_starts]

    @classmethod
    def _combine_cluster(cls, dets):
        """Combine a cluster of detections into one detection."""
        dets_iter = iter(dets)
        x0, y0, x1, y1, confidence = next(dets_iter)
        for dx0, dy0, dx1, dy1, dconfidence in dets_iter:
            x0 = min(x0, dx0)
            y0 = min(y0, dy0)
            x1 = max(x1, dx1)
            y1 = max(y1, dy1)
            confidence = max(confidence, dconfidence)

        return x0, y0, x1, y1, confidence

    @classmethod
    def _combine_box_detections(cls, box1, dets1, box2, dets2, iou_threshold):
        """Combine detections from two boxes.

        Returns a list that contains all detections from dets1 and dets2, some
        of them combined if there's sufficient overlap (i.e. iou for parts of
        detections that are inside of both boxes is >= iou_threshold).

        Note: if a and b are both combined with c, this will cause all three
        to be combined into a single detection. Precisely speaking, each
        equivalence class of the transitive closure of the "overlapping" binary
        relation will be combined into one detection.

        Note: when multiple detections are combined, the combined probability
        is the maximum of the original probabilities.

        """
        box1_over_box2 = utils.intersect(box1, box2)

        # If the two boxes don't overlap, combine nothing...
        if box1_over_box2 is None:
            return dets1 + dets2

        # ...otherwise combine overlapping clusters:
        @functools.lru_cache(maxsize=4096)
        def is_overlap(d1, d2):
            """Does d1 overlap d2 constrained to box1 ^ box2?"""
            d1_o = utils.intersect(d1[:4], box1_over_box2)
            d2_o = utils.intersect(d2[:4], box1_over_box2)
            if d1_o is None or d2_o is None:
                return False
            return utils.iou(d1_o, d2_o) >= iou_threshold

        dets = [
            sorted(dets1, key=lambda d: d[4], reverse=True),
            sorted(dets2, key=lambda d: d[4], reverse=True),
        ]
        picked = {d: False for d in dets[0] + dets[1]}

        def get_cluster(d, side=0):
            """Yield detections overlapping d recursively; mark them picked."""
            picked[d] = True
            yield d
            other_side = 1 - side
            for d_ in dets[other_side]:
                if not picked[d_] and is_overlap(d, d_):
                    yield from get_cluster(d_, other_side)

        return [
            cls._combine_cluster(get_cluster(d))
            for d in dets[0]
            if not picked[d]
        ] + [
            d for d in dets[1]
            if not picked[d]
        ]

    @classmethod
    def _combine_slice_detections(cls, slice_boxes, slice_detections,
                                  iou_threshold):
        """Combine detections from all slices."""
        all_detections = []  # Completed detections outside of next box.
        last_detections = slice_detections[0]
        last_box = slice_boxes[0]

        for detections, box in zip(slice_detections[1:], slice_boxes[1:]):
            detections = [_to_absolute(d, box) for d in detections]
            active = []
            for ld in last_detections:
                if utils.intersect(ld[:4], box) is None:
                    all_detections.append(ld)
                else:
                    active.append(ld)

            last_detections = cls._combine_box_detections(
                last_box, active,
                box, detections,
                iou_threshold,
            )
            last_box = box

        return all_detections + last_detections

    def detect(self, image, path, confidence_threshold=None,
               iou_threshold=None, slicing_threshold=None, slice_overlap=None):
        """Detect ads using wrapped detector and slicing as necessary.

        Parameters
        ----------
        image : PIL.Image
            Source image for ad detection.
        path : str
            Path to the image (it's not used by this detector but is a part of
            detector API).
        confidence_threshold : float
            Minimal confidence for the detection to be counted.
        iou_threshold : float
            Minimal IoU for two detections to be considered duplicated.
        slicing_threshold : float
            Aspect ratio threshold after which input images will be sliced.
        slice_overlap : float
            Percentage of overlap between adjacent slices.

        Returns
        -------
        detections : list of [x0, y0, x1, y1, confidence]
            Detected ad boxes.

        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        if slicing_threshold is None:
            slicing_threshold = self.slicing_threshold
        if slice_overlap is None:
            slice_overlap = self.slice_overlap

        slice_boxes = self._slice_boxes(
            image.size,
            slicing_threshold,
            slice_overlap,
        )
        slices = [
            (
                image.crop(box),
                '{0}_{1[0]},{1[1]}-{1[2]},{1[3]}'.format(path, box),
            )
            for box in slice_boxes
        ]
        slice_detections = [
            detections
            for path, detections in self.detector.batch_detect(
                slices,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )
        ]

        return self._combine_slice_detections(
            slice_boxes=slice_boxes,
            slice_detections=slice_detections,
            iou_threshold=iou_threshold,
        )
