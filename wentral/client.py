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

"""Client for the object detection web service."""

import io
import urllib.parse as urlparse

import requests

import wentral.detector as det


class ProxyDetector(det.Detector):
    """Detector that forwards detection requests to a remote web service.

    Parameters
    ----------
    server_url : str
        URL of the server where object detector web service is running.

    """

    def __init__(self, server_url):
        super().__init__(server_url=server_url)

    def detect(self, image, path, **params):
        """Upload the image for object detection and return the response.

        Parameters
        ----------
        image : PIL.Image or bytes or file
            Source image for object detection.
        path : str
            Path to the image (it's not used by this detector but is a part of
            detector API).
        params : dict
            The rest of the parameters will be passed to the remote server
            without changes.

        Returns
        -------
        detections : list of [x0, y0, x1, y1, confidence]
            Detected boxes.

        """
        if not (isinstance(type(image), type(b'')) or hasattr(image, 'read')):
            bio = io.BytesIO()
            image.save(bio, format='PNG')
            image = bio.getvalue()
        request = requests.post(
            urlparse.urljoin(self.server_url, 'detect'),
            files={'image': (path, image)},
            data=params,
        )
        return [tuple(box) for box in request.json()['boxes']]
