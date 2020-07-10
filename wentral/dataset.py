# This file is part of Wentral
# <https://gitlab.com/eyeo/machine-learning/wentral/>,
# Copyright (C) 2019-present eyeo GmbH
#
# Wentral is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# Wentral is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wentral. If not, see <http://www.gnu.org/licenses/>.

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
    ad_region_types : list of str
        Region types that are considered ads.

    """

    def __init__(self, path):
        self.path = path
        self.index = idx.reg_index(path)
        self.ad_region_types = [
            rt for rt in self.index.region_types
            if 'label' not in rt
        ]
        logging.debug('Ad region types: {}'.format(self.ad_region_types))

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
            Images, their paths and ad boxes.

        """
        for image_name in sorted(self.index):
            image_path = os.path.join(self.path, image_name)
            ad_boxes = [
                region[:4] for region in self.index[image_name]
                if region[4] in self.ad_region_types
            ]
            yield PIL.Image.open(image_path), image_path, ad_boxes


class JsonDataset:
    """A set of images with marked regions loaded from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    index : FragmentIndex
        Index of regions.
    ad_region_types : list of str
        Region types that are considered ads.

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
            Images, their paths and ad boxes.

        """
        for img in self.data['images']:
            image_path = os.path.join(self.images_path, img['image_name'])
            boxes = [gt[:4] for gt in img['ground_truth']]
            yield PIL.Image.open(image_path), image_path, boxes
