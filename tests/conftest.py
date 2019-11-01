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

"""Common testing bits."""

import os
import time

from PIL import Image
import pytest
import wsgi_intercept as icpt
from wsgi_intercept import requests_intercept

import ady.webservice as ws


class MockAdDetector:
    """Ad detector that returns prepared answers."""

    def __init__(self, answers):
        self.answers = answers
        self.log = []
        self.delay = 0

    def detect(self, image, image_path, **kw):
        image_name = os.path.basename(image_path)
        self.log.append({
            'image_name': image_name,
            'params': kw,
        })
        if self.delay > 0:
            time.sleep(self.delay)
        if image_path in self.answers:
            return self.answers[image_path]
        return self.answers.get(image_name, [])

    def __str__(self):
        return 'mock-detector'


@pytest.fixture()
def mock_detector():
    return MockAdDetector({
        '0.png': [
            (0, 0, 50, 20, 0.9),    # Precise detection.
            (82, 12, 94, 45, 0.8),  # Smaller detection.
            (0, 70, 50, 90, 0.7),   # False positive.
        ],
        '1.png': [
            (10, 10, 80, 25, 0.9),  # Precise detection.
            (10, 30, 30, 60, 0.6),  # False positive.
        ],
        '2.png': [
            (5, 5, 15, 15, 0.7),    # Too small: IoU below threshold.
            (20, 20, 50, 50, 0.6),  # Too big: IoU below threshold.
            (60, 60, 90, 90, 0.9),  # Precise detection.
        ],
        'foo.png': [                # For client-server test.
            (10, 20, 30, 40, 0.9),
        ],
    })


@pytest.fixture()
def webservice(mock_detector):
    app = ws.make_app(mock_detector)
    host, port = 'localhost', 8080
    url = 'http://{0}:{1}/'.format(host, port)
    requests_intercept.install()
    icpt.add_wsgi_intercept(host, port, lambda: app)
    yield {'app': app, 'url': url}
    icpt.remove_wsgi_intercept()


@pytest.fixture()
def dataset_dir(tmpdir_factory):
    """Directory with images and marked regions."""
    ret = tmpdir_factory.mktemp('dataset_dir')

    for i in range(3):
        path = ret.join('{}.png'.format(i))
        Image.new('RGB', (100, 100), (0, 0, 0)).save(str(path))

    ret.join('index.csv').write('\n'.join([
        '0.png,0,0,50,20,ad',
        '0.png,10,0,20,5,ad_label',
        '0.png,80,10,95,50,textad',
        '1.png,10,10,80,25,ad',
        '1.png,10,10,15,15,adlabel',
        '2.png,0,0,20,20,ad',
        '2.png,30,30,40,40,ad',
        '2.png,60,60,90,90,textad',
    ]))

    return ret
