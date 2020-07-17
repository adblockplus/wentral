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

"""Ad detector that reads detections from a region index."""

import os

import wentral.ad_detector as ad
import wentral.dataset as ds


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
