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
import time
from timeit import default_timer as timer

import flask
import PIL
import paste.translogger as tl
import psutil
import waitress

from ady.detector import AdDetector

app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return """
<html>
  <body>
    <form action="/detect" method="POST" enctype="multipart/form-data">
      <input type="file" name="image" />
      <input type="submit" value="submit" name="submit" />
    </form>
  </body>
</html>
"""


@app.route('/detect', methods=['POST'])
def detect():
    image_file = flask.request.files['image']
    image = PIL.Image.open(image_file)

    process = psutil.Process(os.getpid())
    logging.debug('RSS before detection: %d', process.memory_info().rss)
    logging.debug('Detecting ads')
    t1 = timer()
    boxes = app.detector.detect(image)
    t2 = timer()
    logging.info('Found {} ads in {} seconds'.format(len(boxes), t2 - t1))
    logging.debug('RSS after detection: %d', process.memory_info().rss)

    response_body = json.dumps({
        'size': image.size,
        'boxes': boxes,
        'detection_time': t2 - t1,
    })
    response_headers = {
        'Content-type': 'application/json',
    }
    return response_body, response_headers


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    weights_file = os.environ['YOLOv3_WEIGHTS_PATH']
    app.detector = AdDetector(weights_file)
    waitress.serve(tl.TransLogger(app, setup_console_handler=False),
                   listen='*:8080')


if __name__ == '__main__':
    main()
