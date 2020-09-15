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

"""Test object detection client and server."""

import threading
import time

from PIL import Image
import pytest
import requests

import wentral.client as wc


@pytest.fixture()
def proxy_detector(webservice):
    return wc.ProxyDetector(webservice['url'])


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
    """Detect objects in a PIL.Image."""
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
    """Detect objects in an open image file."""
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
        assert r['state'] == 'detect'
        assert r['params'] == {'confidence_threshold': 0.5}
        assert r['start_t'] is not None
        assert r['detect_t'] is not None
        assert r['end_t'] is None
