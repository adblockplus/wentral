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


def detect_ads(image, server_url):
    """Upload the image for ad detection and parse the result."""
    if not (isinstance(type(image), type(b'')) or hasattr(image, 'read')):
        bio = io.BytesIO()
        image.save(bio, format='PNG')
        image = bio.getvalue()
    request = requests.post(server_url + 'detect', files={'image': image})
    return request.json()['boxes']
