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

"""AdDetector base class and API definition."""

from typing import List, Iterable, Tuple

from PIL.Image import Image


# Detection is (x0, y0, x1, y1, confidence):
Detection = Tuple[float, float, float, float, float]

# Detection list is all detections for an image:
DetectionList = List[Detection]

# Detections for multiple images (prefixed by their paths):
DetectionLists = Iterable[Tuple[str, List[Detection]]]


class AdDetector:
    """Ad detector encapsulates ad detection logic of any kind.

    See `detect` and `batch_detect`.

    """

    def __init__(self, **params):
        """Implementations should pass detector parameters here.

        Parameters passed to this constructor will be assigned to object
        attributes with the same names and will be visible in the output of
        __str__.

        When defining ad detectors, try to provide sensible defaults for as
        many parameters as possible. This makes it easier to use them without
        knowning what the sensible values would be.

        """
        self.param_names = sorted(params)
        for k, v in params.items():
            setattr(self, k, v)

    def __str__(self):
        """Return a string that contains detector name and parameters.
        """
        name = self.__class__.__name__
        params = ', '.join(
            '{}={}'.format(pn, str(getattr(self, pn)))
            for pn in self.param_names
        )
        return '{}({})'.format(name, params)

    def detect(self, image: Image, path: str, **params) -> DetectionList:
        """Detect ads in the image, return all detected boxes as a list.

        Implementations can accept additional keyword arguments.

        Parameters
        ----------
        image : PIL.Image
            Source image for ad detection.
        path : str
            Path to the image (it's not used by this detector but is a part of
            detector API).
        params
            Additional parameters, to override the defaults set in the
            constructor.

        Returns
        -------
        detections : list of (x0, y0, x1, y1, confidence)
            Detected ad boxes.

        """
        raise NotImplementedError()

    def batch_detect(self, images: List[Tuple[Image, str]],
                     **params) -> DetectionLists:
        """Detect ads in multiple images (possibly in parralel).

        The return value might be a generator (this allows implementations to
        return detections as soon as they are ready).

        Note: default implementation delegates to `detect`.

        Parameters
        ----------
        images : list of (PIL.Image, str)
            List of tuples containing images and their paths.
        params
            Additional parameters, to override the defaults set in the
            constructor.

        Returns
        -------
        detections : iterable of (str, list of (x0, y0, x1, y1, confidence))
            Detected ad boxes (the sequence doesn't have to be the same as the
            original sequence of images.

        """
        for image, path in images:
            yield (path, self.detect(image, path, **params))
