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
        self.url = url

    def __str__(self):
        return 'ProxyAdDetector({})'.format(self.url)

    def detect(self, image, path):
        """Upload the image for ad detection and return the response."""
        if not (isinstance(type(image), type(b'')) or hasattr(image, 'read')):
            bio = io.BytesIO()
            image.save(bio, format='PNG')
            image = bio.getvalue()
        request = requests.post(
            self.url + 'detect',
            files={'image': (path, image)},
        )
        return [tuple(box) for box in request.json()['boxes']]
