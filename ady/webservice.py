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

"""Flask-based web service that detects ads in screenshots."""

import json
import logging
import os
from timeit import default_timer as timer
import threading

import flask
import PIL
import psutil


class RequestData:
    """Information about a request to the web service."""

    # State constants.
    PREPARE = 'prepare'
    AD_DETECT = 'ad-detect'
    RESPONSE = 'response'

    def __init__(self, request_id):
        self.id = request_id
        self.state = self.PREPARE
        self.image_name = None
        self.params = None
        self.start_t = timer()
        self.detect_t = None
        self.end_t = None

    def to_detect(self):
        self.state = self.AD_DETECT
        self.detect_t = timer()

    def to_response(self):
        self.state = self.RESPONSE
        self.end_t = timer()

    def to_dict(self):
        return self.__dict__


def _mem_rss():
    """Return resident memory size in bytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


class Counter:
    """Thread-safe counter."""

    value = 0

    def __init__(self):
        self.lock = threading.Lock()

    def next(self):
        """Return next value of the counter."""
        self.lock.acquire()
        try:
            return self.value
        finally:
            self.value += 1
            self.lock.release()


def make_app(detector):
    """Make a Flask-based web-service that detects ads using `detector`."""
    app = flask.Flask(__name__)
    app.detector = detector
    app.id_counter = Counter()
    app.requests = {}

    @app.route('/', methods=['GET'])
    def index():
        return """
    <html>
      <body>
        <form action="/detect" method="POST" enctype="multipart/form-data">
          <p>
            Image<br/>
            <input type="file" name="image" />
          </p>
          <p>
            Confidence threshold<br/>
            <input type="text" name="confidence_threshold" value="0.5" />
          </p>
          <p>
            IoU threshold<br/>
            <input type="text" name="iou_threshold" value="0.4" />
          </p>
          <input type="submit" value="submit" name="submit" />
        </form>
        <a href="/status">server status</a>
      </body>
    </html>
    """

    @app.route('/status')
    def status():
        """Return status information as JSON."""
        return {
            'mem_rss': _mem_rss(),
            'detector': str(app.detector),
            'requests': [r.to_dict() for r in app.requests.values()],
        }

    @app.route('/detect', methods=['POST'])
    def detect():
        """Detect ads in uploaded image."""
        request_id = app.id_counter.next()
        request_data = app.requests[request_id] = RequestData(request_id)

        try:
            image_file = flask.request.files['image']
            image_name = request_data.image_name = image_file.filename
            image = PIL.Image.open(image_file)

            kw = request_data.params = {}
            for x in ['confidence_threshold', 'iou_threshold']:
                if x in flask.request.form:
                    try:
                        kw[x] = float(flask.request.form[x])
                    except ValueError:
                        flask.abort(400, '{} must be a number'.format(x))

            logging.debug('Got request: {} {}'.format(image_name, kw))
            logging.debug('RSS before detection: %d', _mem_rss())
            request_data.to_detect()
            boxes = app.detector.detect(image, image_name, **kw)
            request_data.to_response()
            det_time = request_data.end_t - request_data.detect_t
            logging.info('Found {} ads in {} seconds'.format(len(boxes),
                                                             det_time))
            logging.debug('RSS after detection: %d', _mem_rss())

            response_body = json.dumps({
                'size': image.size,
                'boxes': boxes,
                'detection_time': det_time,
            })
            response_headers = {
                'Content-type': 'application/json',
            }
            return response_body, response_headers
        finally:
            del app.requests[request_id]

    return app
