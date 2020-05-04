# This file is part of Ad Detect YOLO <https://adblockplus.org/>,
# Copyright (C) 2019-present eyeo GmbH
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

"""Ad detector that reads detections from a region index."""

import os

import ady.ad_detector as ad
import ady.dataset as ds


class StaticDetector(ad.AdDetector, ds.LabeledDataset):
    """Detector based on a region index.

    Parameters
    ----------
    path : str
        Path to the directory that contains the region index.

    """

    def __init__(self, path):
        ad.AdDetector.__init__(self, path=path)
        ds.LabeledDataset.__init__(self, path)

    def detect(self, image, image_path, **kw):
        """Return ad detections based on marked regions.

        Raises
        ------
        KeyError
            If there are no regions for specific image in this dataset.

        """
        image_name = os.path.basename(image_path)
        try:
            regions = self.index[image_name]
        except KeyError:
            raise Exception('Regions information is missing for {} in {}'
                            .format(image_name, self.path))

        return [
            region[:4] + (0.999,) for region in regions
            if region[4] in self.ad_region_types
        ]
