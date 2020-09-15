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

"""Flask-based web service that detects objects in screenshots."""

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
    DETECT = 'detect'
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
        self.state = self.DETECT
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
    """Make a Flask-based web-service that detects objects using `detector`."""
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
        """Detect objects in uploaded image."""
        request_id = app.id_counter.next()
        request_data = app.requests[request_id] = RequestData(request_id)

        try:
            image_file = flask.request.files['image']
            image_name = request_data.image_name = image_file.filename
            image = PIL.Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')

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
            logging.info('Found {} objects in {} seconds'.format(len(boxes),
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
