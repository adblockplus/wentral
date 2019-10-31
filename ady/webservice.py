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

import flask
import PIL
import psutil


def make_app(detector):
    """Make a Flask-based web-service that detects ads using `detector`."""
    app = flask.Flask(__name__)
    app.detector = detector

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
      </body>
    </html>
    """

    @app.route('/detect', methods=['POST'])
    def detect():
        image_file = flask.request.files['image']
        image_name = image_file.filename
        image = PIL.Image.open(image_file)

        kw = {}
        for x in ['confidence_threshold', 'iou_threshold']:
            if x in flask.request.form:
                try:
                    kw[x] = float(flask.request.form[x])
                except ValueError:
                    flask.abort(400, '{} must be a number'.format(x))

        process = psutil.Process(os.getpid())
        logging.debug('Got request: {} {}'.format(image_name, kw))
        logging.debug('RSS before detection: %d', process.memory_info().rss)
        t1 = timer()
        boxes = app.detector.detect(image, image_name, **kw)
        t2 = timer()
        logging.info('Found {} ads in {} seconds'.format(len(boxes), t2 - t1))
        logging.debug('RSS after detection: %d', process.memory_info().rss)
        det_time = t2 - t1

        response_body = json.dumps({
            'size': image.size,
            'boxes': boxes,
            'detection_time': det_time,
        })
        response_headers = {
            'Content-type': 'application/json',
        }
        return response_body, response_headers

    return app
