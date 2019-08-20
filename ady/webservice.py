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
import numpy as np
import paste.translogger as tl
from PIL import Image
import waitress

from ady.yolo_v3 import non_max_suppression
from ady.utils import *


# Size of the input image for the detector.
YOLO_SIZE = 416

# Detection parameters
CONF_THRESHOLD = 0.5  # Level of confidence that we count as detection.
IOU_THRESHOLD = 0.4   # IOU above which two boxes are considered the same.

# Region type (a.k.a. class) that means "advertisement".
AD_TYPE = 0


def scale_box(box, img_size):
    """Scale detected box to match image size."""
    xscale = img_size[0] / YOLO_SIZE
    yscale = img_size[1] / YOLO_SIZE
    x0, y0, x1, y1 = box
    return [
        float(x0) * xscale,
        float(y0) * yscale,
        float(x1) * xscale,
        float(y1) * yscale,
    ]


class AdDetector:
    """Ad detector that encapsulates TF session and detection model."""

    def __init__(self, weights_file):
        classes = {AD_TYPE: 'ad'}
        self.inputs = tf.placeholder(tf.float32,
                                     [None, YOLO_SIZE, YOLO_SIZE, 3])
        config = tf.ConfigProto()
        logging.info('Initializing TF session')
        self.tf_session = tf.Session(config=config)
        logging.info('Initializing YOLOv3 and loading weights from %s',
                     weights_file)
        self.detections, self.boxes = init_yolo(
            self.tf_session, self.inputs, len(classes),
            weights_file, header_size=4,
        )
        logging.info('Done')

    def detect(self, image):
        """Detect ads in the image, return detection results as a dict.

        The return value is as follows:

            {
                'size': [image_width, image_height],
                'boxes': [
                    [x0, y0, x1, y1, probability],
                    ...
                ],
            }

        """
        img = image.resize((YOLO_SIZE, YOLO_SIZE))
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')

        logging.info('Detecting ads')
        t1 = timer()
        detected_boxes = self.tf_session.run(
            self.boxes,
            feed_dict={self.inputs: [np.array(img, dtype=np.float32)]},
        )
        unique_boxes = non_max_suppression(
            detected_boxes,
            confidence_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
        )
        boxes = [scale_box(box, image.size) + [float(p)]
                 for box, p in unique_boxes[AD_TYPE]]
        t2 = timer()
        logging.debug('Detected boxes: {}'.format(boxes))
        logging.info('Detection complete: found {} ads in {} seconds'
                     .format(len(boxes), t2 - t1))

        return {
            'size': image.size,
            'boxes': boxes,
            'detection_time': t2 - t1,
        }


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
    image = Image.open(image_file)
    response_body = json.dumps(app.detector.detect(image))
    response_headers = {
        'Content-type': 'application/json',
    }
    return response_body, response_headers


def serve(argv):
    app.detector = AdDetector('../ad-versarial/models/page_based_yolov3.weights')
    waitress.serve(tl.TransLogger(app, setup_console_handler=False),
                   listen='*:8080')


def main():
    logging.basicConfig(level=logging.INFO)
    serve([])
    # tf.app.run(main=serve)


if __name__ == '__main__':
    main()
