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

"""Client for the ad detection web service."""

import io
import urllib.parse as urlparse

import requests

import ady.ad_detector as ad


class ProxyAdDetector(ad.AdDetector):
    """Ad detector that forwards detection requests to a remote web service.

    Parameters
    ----------
    server_url : str
        URL of the server where ad detector web service is running.

    """

    def __init__(self, server_url):
        super().__init__(server_url=server_url)

    def detect(self, image, path, **params):
        """Upload the image for ad detection and return the response.

        Parameters
        ----------
        image : PIL.Image or bytes or file
            Source image for ad detection.
        path : str
            Path to the image (it's not used by this detector but is a part of
            detector API).
        params : dict
            The rest of the parameters will be passed to the remote server
            without changes.

        Returns
        -------
        detections : list of [x0, y0, x1, y1, confidence]
            Detected ad boxes.

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
