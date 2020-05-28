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

import threading
import time

from PIL import Image
import pytest
import requests

import ady.client as wc


@pytest.fixture()
def proxy_detector(webservice):
    return wc.ProxyAdDetector(webservice['url'])


@pytest.fixture()
def get_server_status(webservice):
    def get_server_status():
        request = requests.get(webservice['url'] + '/status')
        return request.json()
    return get_server_status


@pytest.fixture()
def screenshot_image():
    return Image.new('RGBA', (100, 100), '#123456')


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


def test_server_status(get_server_status):
    """Test server status with no requests."""
    status = get_server_status()
    assert status['requests'] == []
    assert status['detector'] == 'mock-detector'
    assert status['mem_rss'] > 1000


def test_server_status_req(mock_detector, get_server_status, proxy_detector,
                           screenshot_image):
    """Test server status while sending requests."""
    mock_detector.delay = 0.1
    for i in range(3):
        threading.Thread(
            target=proxy_detector.detect,
            args=(screenshot_image, '{}.png'.format(i)),
            kwargs={'confidence_threshold': 0.5},
            daemon=True,
        ).start()
    time.sleep(0.05)
    status = get_server_status()
    reqs = status['requests']
    assert len(reqs) == 3
    assert set(r['image_name'] for r in reqs) == {'0.png', '1.png', '2.png'}
    assert set(r['id'] for r in reqs) == {0, 1, 2}
    for r in reqs:
        assert r['state'] == 'ad-detect'
        assert r['params'] == {'confidence_threshold': 0.5}
        assert r['start_t'] is not None
        assert r['detect_t'] is not None
        assert r['end_t'] is None
