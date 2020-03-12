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

"""YOLOv3-based ad detector."""

import collections
import logging
import os
import warnings

import numpy as np

import ady.ad_detector as ad
import ady.utils as u

# Default size of the input image for the detector.
YOLO_SIZE = 416

# Detection parameters
CONF_THRESHOLD = 0.5     # Level of confidence that we count as detection.
IOU_THRESHOLD = 0.4      # IOU above which two boxes are considered the same.

# Region type (a.k.a. class) that means "advertisement".
AD_TYPE = 0


def deduplicate(detections, iou_threshold=0.4):
    """Remove overlapping detections (leave maximal confidence one).

    Parameters
    ----------
    detections : np.array(-1, 5 + n_cls)
        Detected boxes. In the last dimension the values are box coordinates
        followed by confidence and class probabilities.
    iou_threshold : float
        Minimal IoU at which we consider that two boxes are overlapping.

    Returns
    -------
    class_map : dict(clss -> [(box, score)])
        Map of class ids to detections of the class.

    """
    classes = np.argmax(detections[:, 5:], axis=-1)
    result = collections.defaultdict(list)

    for cls in set(classes):
        cls_all = detections[classes == cls]
        cls_all = cls_all[cls_all[:, 4].argsort()[::-1]]
        while len(cls_all) > 0:
            first = cls_all[0]
            result[cls].append((first[:4], first[4]))
            cls_all = cls_all[1:]
            ious = np.array([
                u.iou(first[:4], x)
                for x in cls_all[:, :4]
            ])
            cls_all = cls_all[ious < iou_threshold]

    return result


class YoloAdDetector(ad.AdDetector):
    """Ad detector that encapsulates TF session and YOLO v.3 model."""

    def __init__(self, weights_file, confidence_threshold=CONF_THRESHOLD,
                 iou_threshold=IOU_THRESHOLD):
        super().__init__(
            weights_file=weights_file,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        self._detect_yolo_params()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._init_yolo()

    def _detect_yolo_params(self):
        """Autodetect model parameters based on the size of the weights file.

        First we detect how many object classes we have based on 61570957 float
        parameters that are not class-related and 5385 class-related ones. Then
        we calculate header size as the remaining size of the weights file.

        Note: the approach above is empirically derived from two existing
        weights files and might be completely wrong.

        """
        MAIN_WEIGHTS = 61570957
        CLASS_WEIGHTS = 5385
        size = os.stat(self.weights_file).st_size
        words = size // 4
        self.class_count = (words - MAIN_WEIGHTS) // CLASS_WEIGHTS
        self.header_size = (words - MAIN_WEIGHTS
                            - CLASS_WEIGHTS * self.class_count)
        logging.debug('Detected params: CC: %d HS: %d',
                      self.class_count, self.header_size)
        if self.header_size not in {4, 5}:
            raise Exception('Expected header_size to be 4 or 5, not {}'
                            .format(self.header_size))

    def _init_yolo(self):
        """Create YOLO graph and load the weights."""
        logging.debug('Import TF and YOLO model')
        # Importing TensorFlow and YOLO produces lots of warnings that get in
        # the way and are out of our control so we silence them.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tensorflow
            import ady.yolo.yolo_v3 as yolo
            tf = tensorflow.compat.v1

        logging.debug('Create TF session')
        self.tf_session = tf.Session()
        logging.debug('Building YOLO graph')
        self.inputs = tf.placeholder(tf.float32,
                                     [None, YOLO_SIZE, YOLO_SIZE, 3])
        with tf.variable_scope('detector'):
            detections = yolo.yolo_v3(self.inputs, self.class_count,
                                      data_format='NHWC')
            logging.debug('Loading weights from %s', self.weights_file)
            load_ops = yolo.load_weights(
                tf.global_variables(scope='detector'),
                self.weights_file,
                header_size=self.header_size,
            )
        self.boxes = yolo.detections_boxes(detections)
        logging.debug('Applying weights to the graph')
        self.tf_session.run(load_ops)
        logging.debug('Done')

    def scale_box(self, box, img_size):
        """Scale detected box to match image size."""
        xscale = img_size[0] / YOLO_SIZE
        yscale = img_size[1] / YOLO_SIZE
        x0, y0, x1, y1 = map(float, box)
        return [x0 * xscale, y0 * yscale, x1 * xscale, y1 * yscale]

    def _get_detections(self, image):
        """Run the model and return detections as np.array(-1, 5 + n_cls)."""
        all_detections = self.tf_session.run(
            self.boxes,
            feed_dict={self.inputs: [np.array(image, dtype=np.float32)]},
        )
        return all_detections.reshape(-1, all_detections.shape[-1])

    def detect(self, image, path, confidence_threshold=None,
               iou_threshold=None):
        """Detect ads in the image.

        Parameters
        ----------
        image : PIL.Image
            Source image for ad detection.
        path : str
            Path to the image (it's not used by this detector but is a part of
            detector API).
        confidence_threshold : float
            Minimal confidence for the detection to be counted.
        iou_threshold : float
            Minimal IoU for two detections to be considered duplicated.

        Returns
        -------
        detections : list of (x0, y0, x1, y1, confidence)
            Detected ad boxes.

        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        img = image.resize((YOLO_SIZE, YOLO_SIZE))
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')

        detections = self._get_detections(img)
        # Discard low confidence detections.
        detections = detections[detections[:, 4] > confidence_threshold]
        logging.debug('Detected %d boxes: %s', len(detections), detections)
        unique_boxes = deduplicate(detections, iou_threshold=iou_threshold)
        logging.debug('Unique boxes: %s', unique_boxes)

        return [
            tuple(self.scale_box(box, image.size)) + (float(p),)
            for class_id in range(self.class_count)
            for box, p in unique_boxes.get(class_id, [])
        ]
