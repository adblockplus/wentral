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

"""Client for the ad detection web service."""

import io

import requests


class ProxyAdDetector:
    """Ad detector that forwards detection requests to a remote web service."""

    def __init__(self, url):
        if not url.endswith('/'):
            url += '/'
        self.url = url

    def __str__(self):
        return 'ProxyAdDetector({})'.format(self.url)

    def detect(self, image, path, confidence_threshold=None,
               iou_threshold=None):
        """Upload the image for ad detection and return the response.

        Parameters
        ----------
        image : PIL.Image or bytes or file
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
        params = {}
        if confidence_threshold is not None:
            params['confidence_threshold'] = str(confidence_threshold)
        if iou_threshold is not None:
            params['iou_threshold'] = str(iou_threshold)

        if not (isinstance(type(image), type(b'')) or hasattr(image, 'read')):
            bio = io.BytesIO()
            image.save(bio, format='PNG')
            image = bio.getvalue()
        request = requests.post(
            self.url + 'detect',
            files={'image': (path, image)},
            data=params,
        )
        return [tuple(box) for box in request.json()['boxes']]
