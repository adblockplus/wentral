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

"""Common testing bits."""

import os
import time

from PIL import Image
import pytest
import wsgi_intercept as icpt
from wsgi_intercept import requests_intercept

import wentral.ad_detector as ad
import wentral.webservice as ws
import wentral.benchmark as bm
import wentral.dataset as ds


class MockAdDetector(ad.AdDetector):
    """Ad detector that returns prepared answers."""

    name = 'mock-detector'

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
        return self.name


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
    """Mock ad detection web service."""
    app = ws.make_app(mock_detector)
    host, port = 'localhost', 8080
    url = 'http://{0}:{1}/'.format(host, port)
    requests_intercept.install()
    icpt.add_wsgi_intercept(host, port, lambda: app)
    yield {'app': app, 'url': url}
    icpt.remove_wsgi_intercept()


@pytest.fixture()
def shmetector(mocker, mock_detector):
    """Mock detector that can be imported into wbm and wws with -d."""
    fqn = 'wentral.Shmetector'

    def construct_detector(weights_file, iou_threshold=0.5, missing=None,
                           extra_one=None, extra_two=None):
        """Mock of detector constructor. Note: it requires `weights_file`."""
        assert weights_file == '/a/b/c'
        if extra_two is not None:
            print('extra_two=' + extra_two)
        if extra_one is not None:
            print('extra_one=' + extra_one)
        mock_detector.name = ('MD(weights_file={}, iou_threshold={})'
                              .format(weights_file, iou_threshold))
        return mock_detector

    mocker.patch(fqn, construct_detector, create=True)
    return fqn


@pytest.fixture()
def dataset_dir(tmpdir_factory):
    """Directory with images and marked regions."""
    ret = tmpdir_factory.mktemp('dataset_dir')

    for i in range(3):
        path = ret.join('{}.png'.format(i))
        mode = 'RGB' if i == 0 else 'RGBA'
        Image.new(mode, (100, 100), (0, 0, 0)).save(str(path))

    ret.join('index.csv').write('\n'.join([
        'image,xmin,ymin,xmax,ymax,label',
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


@pytest.fixture()
def dataset(dataset_dir):
    """Dataset loaded from `dataset_dir`."""
    return ds.LabeledDataset(str(dataset_dir))


@pytest.fixture()
def json_output(dataset, mock_detector, tmpdir):
    """JSON file with an output of a run."""
    evaluation = bm.evaluate(dataset, mock_detector)
    json_file = tmpdir.join('output.json')
    with json_file.open('wt', encoding='utf-8') as jf:
        evaluation.json_dump(jf)
    return json_file
