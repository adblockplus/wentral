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

from PIL import Image
import pytest

import ady.client as wc


@pytest.fixture()
def proxy_detector(webservice):
    return wc.ProxyAdDetector(webservice['url'])


@pytest.fixture()
def screenshot_image():
    return Image.new('RGB', (100, 100), '#123456')


def test_client_server(proxy_detector, screenshot_image, webservice):
    """Detect ads in a PIL.Image."""
    boxes = proxy_detector.detect(screenshot_image, 'foo.png')
    log = webservice['app'].detector.log
    assert boxes == [(10, 20, 30, 40, 0.9)]
    assert log == [{'image_name': 'foo.png', 'params': {}}]


def test_client_server_params(proxy_detector, screenshot_image, webservice):
    proxy_detector.detect(screenshot_image, '0.png', confidence_threshold=0.7)
    proxy_detector.detect(screenshot_image, '0.png', iou_threshold=0.6)
    proxy_detector.detect(screenshot_image, '0.png', confidence_threshold=0.3,
                          iou_threshold=0.2)
    log = webservice['app'].detector.log
    assert log == [
        {'image_name': '0.png', 'params': {'confidence_threshold': 0.7}},
        {'image_name': '0.png', 'params': {'iou_threshold': 0.6}},
        {'image_name': '0.png', 'params': {'confidence_threshold': 0.3,
                                           'iou_threshold': 0.2}},
    ]


def test_client_server_file(proxy_detector, screenshot_image, tmpdir):
    """Detect ads in an open image file."""
    img_path = tmpdir.join('foo.png')
    screenshot_image.save(str(img_path))
    with img_path.open('rb') as im_file:
        boxes = proxy_detector.detect(im_file, 'foo.png')
    assert boxes == [(10, 20, 30, 40, 0.9)]
