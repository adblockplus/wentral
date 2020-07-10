# This file is part of Wentral
# <https://gitlab.com/eyeo/machine-learning/wentral/>,
# Copyright (C) 2019-present eyeo GmbH
#
# Wentral is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# Wentral is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wentral. If not, see <http://www.gnu.org/licenses/>.

"""AdDetector that loads the detections from a JSON file."""

import json
import os

import wentral.ad_detector as ad
import wentral.constants as const
import wentral.utils as u


class JsonDetector(ad.AdDetector):
    """AdDetector that loads detections from a JSON file.

    It can still perform additional confidence thresholding and IoU
    deduplication.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    confidence_threshold : float
        Minimal detection confidence.
    iou_threshold : float
        IoU (intersection over union) level at which two detections are
        considered duplicates.

    """

    def __init__(self, path, confidence_threshold=const.CONF_THRESHOLD,
                 iou_threshold=const.IOU_THRESHOLD):
        super().__init__(
            path=path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        self._load_data()

    def _load_data(self):
        """Load detections from a JSON file."""
        with open(self.path, 'rt', encoding='utf-8') as jf:
            data = json.load(jf)

        self.detections = {}
        for img_data in data['images']:
            name = img_data['image_name']
            detections = img_data['detections']
            self.detections[name] = [d[:5] for d in detections]

    def detect(self, image, path, confidence_threshold=None,
               iou_threshold=None):
        """Return detections loaded from JSON file.

        If `confidence_threshold` and/or `iou_threshold` are provided here or
        in the constructor, some detections might be filtered out.

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

        Returns
        -------
        detections : list of [x0, y0, x1, y1, confidence]
            Detected ad boxes.

        """
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        image_name = os.path.basename(path)
        if image_name not in self.detections:
            raise KeyError('No detections data for ' + image_name)

        detections = sorted([
            d for d in self.detections[image_name]
            if d[4] >= confidence_threshold
        ], key=lambda d: d[4], reverse=True)

        picked = []
        for d in detections:
            for p in picked:
                if u.iou(p[:4], d[:4]) >= iou_threshold:  # duplicate.
                    break
            else:
                picked.append(d)

        return picked
