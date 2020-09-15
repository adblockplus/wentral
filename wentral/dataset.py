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

"""Dataset loading for benchmarking."""

import json
import logging
import os

import admincer.index as idx
import PIL


class LabeledDataset:
    """A set of images with marked regions loaded from a directory.

    Attributes
    ----------
    path : str
        Path to the images and region files.
    index : FragmentIndex
        Index of regions.
    region_types : list of str
        Region types that we are interested in. The types in this list will be
        considered detections and the rest will be ignored. By default all
        regions that don't contains the substring "label" are detected.

    """

    def __init__(self, path):
        self.path = path
        self.index = idx.reg_index(path)
        self.region_types = [
            rt for rt in self.index.region_types
            if 'label' not in rt
        ]
        logging.debug('Region types: {}'.format(self.region_types))

    @property
    def images_path(self):
        return self.path

    def __str__(self):
        return 'LabeledDataset(path={})'.format(self.path)

    def __iter__(self):
        """Yield images, paths and marked boxes.

        Yields
        ------
        image_data : (Image, set, list of tuple)
            Images, their paths and detection boxes.

        """
        for image_name in sorted(self.index):
            image_path = os.path.join(self.path, image_name)
            boxes = [
                region[:4] for region in self.index[image_name]
                if region[4] in self.region_types
            ]
            yield PIL.Image.open(image_path), image_path, boxes


class JsonDataset:
    """A set of images with marked regions loaded from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    """

    def __init__(self, path):
        self.path = path
        with open(self.path, 'rt', encoding='utf-8') as jf:
            self.data = json.load(jf)
        self.images_path = self.data['images_path']

    def __str__(self):
        return 'JsonDataset(path={})'.format(self.path)

    def __iter__(self):
        """Yield images, paths and marked boxes.

        Yields
        ------
        image_data : (Image, set, list of tuple)
            Images, their paths and detection boxes.

        """
        for img in self.data['images']:
            image_path = os.path.join(self.images_path, img['image_name'])
            boxes = [gt[:4] for gt in img['ground_truth']]
            yield PIL.Image.open(image_path), image_path, boxes
