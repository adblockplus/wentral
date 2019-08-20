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

import logging

import numpy as np
import tensorflow as tf

import ady.yolo_v3 as yolo


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
    x0, y0, x1, y1 = map(float, box)
    return [x0 * xscale, y0 * yscale, x1 * xscale, y1 * yscale]


class AdDetector:
    """Ad detector that encapsulates TF session and YOLO v.3 model."""

    def __init__(self, weights_file):
        classes = {AD_TYPE: 'ad'}
        self.inputs = tf.placeholder(tf.float32,
                                     [None, YOLO_SIZE, YOLO_SIZE, 3])
        logging.info('Initializing TF session')
        config = tf.ConfigProto()
        self.tf_session = tf.Session(config=config)
        logging.info('Booting YOLO and loading weights from %s', weights_file)
        self.detections, self.boxes = yolo.init(
            self.tf_session, self.inputs, len(classes),
            weights_file, header_size=4,
        )
        logging.info('Done')

    def detect(self, image):
        """Detect ads in the image, return all detected boxes as a list."""
        img = image.resize((YOLO_SIZE, YOLO_SIZE))
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')

        detected_boxes = self.tf_session.run(
            self.boxes,
            feed_dict={self.inputs: [np.array(img, dtype=np.float32)]},
        )
        unique_boxes = yolo.non_max_suppression(
            detected_boxes,
            confidence_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
        )
        return [
            scale_box(box, image.size) + [float(p)]
            for box, p in unique_boxes[AD_TYPE]
        ]
