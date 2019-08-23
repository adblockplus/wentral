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

"""Test ad detection client and server."""

import PIL
import pytest
import wsgi_intercept as icpt
from wsgi_intercept import requests_intercept

import ady.client as wc
import ady.webservice as ws


class FakeDetector:
    """Fake ad detector."""

    def detect(self, image):
        assert image.size == (100, 200)
        return [(10, 20, 30, 40, 0.9)]


@pytest.fixture()
def webservice():
    app = ws.app
    app.detector = FakeDetector()
    host, port = 'localhost', 8080
    url = 'http://{0}:{1}/'.format(host, port)
    requests_intercept.install()
    icpt.add_wsgi_intercept(host, port, lambda: app)
    yield {'app': app, 'url': url}
    icpt.remove_wsgi_intercept()


@pytest.fixture()
def screenshot_image():
    return PIL.Image.new('RGB', (100, 200), '#123456')


def test_client_server(webservice, screenshot_image):
    """Detect ads in a PIL.Image."""
    boxes = wc.detect_ads(screenshot_image, webservice['url'])
    assert boxes == [[10, 20, 30, 40, 0.9]]  # Tuple becomes a list in JSON.


def test_client_server_file(webservice, screenshot_image, tmpdir):
    """Detect ads in an open image file."""
    img_path = tmpdir.join('foo.png')
    screenshot_image.save(str(img_path))
    with img_path.open('rb') as im_file:
        boxes = wc.detect_ads(im_file, webservice['url'])
    assert boxes == [[10, 20, 30, 40, 0.9]]  # Tuple becomes a list in JSON.
