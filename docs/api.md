# Wentral API

This document describes parts of Wentral API that is related to implementing
detectors as well as the web service HTTP API and the Python library for
working with it.

## Implementing detectors

Detector API is specified in `wentral.detector`. This module also provides
a base class for detectors (although detector implementations are not required
to inherit it as long as they implement the API). Basically detectors are
expected to do three things:

- Accept detector parameters as arguments to `__init__`. The most common
  parameters are `weights_file` -- a path to the file containing the model
  weights, `confidence_threshold` and `iou_threshold`. There are no particular
  constraints on these parameters other than that it's convenient to use
  standard names provided by Wentral because it makes it easier to pass them
  from the command line.
  - If a argument of `__init__` has type signature, it will be used by Wentral
    to convert the argument value before passing it to the constructor.
- Implement reasonable `__str__` method that includes any important parameters.
- Implement `detect` and `batch_detect` methods (`Detector` base class provides
  an implementation of `batch_detect` that calls `detect` in a loop but
  implementations are encouraged to do `batch_detect` in parallel when
  possible).

### `detect` method

This method takes one image and returns regions detected in it. The arguments
of `detect` are:

- `image` -- `PIL.Image` object that contains the image for region detection.
- `path` -- Path/name of the image. Some detectors (e.g.
  [json and static](cli.md#special-detectors)) use the path instead of the
  actual image but ML models typically ignore it.

It can also take additional arguments that override detector parameters for
this particular call, such as `confidence_threshold` and `iou_threshold`.

The return value of `detect` should be a list of tuples that contains box
coordinates and detection confidence.

### `batch_detect` method

Batch detect does the same for a batch of images. It takes a list of tuples of
`image` and `path` plus same additional arguments as `detect`.

The return value should be an iterator over `(path, detections)` tuples where
`path` is the original image path and `detections` is a list of detections in
the same format as for `detect`. The order of the return value doesn't have to
be the same as the order of the input. Depending on implementation details
this might be a list or a generator: Wentral is agnostic about this to give
flexibility to the detector implementation.

## Using the web service

Client code for Python is provided in `wentral.client`:

    from PIL import Image
    import wentral.client as cl

    detector = cl.ProxyDetector('http://localhost:8080/')
    path = 'path/image.png'
    image = Image.open(path)
    boxes = detector.detect(image, path)

To use the web service directly, upload the image to `http://host:port/detect`
using a POST request with content type `multipart/form-data`. The field name
should be called `image`. The result is returned as a JSON document that
contains an object with the following keys:

- `size` -- An array with width and height of the image.
- `detection_time` -- Detection time in seconds.
- `boxes` -- Array of arrays that contain detection box coordinates and
  detection confidence.

The requests to `detect` endpoint can also include additional parameters for
the detection process (they are sent as URL parameters):

- `confidence_threshold` - minimum model confidence for detections to be
  returned (default: 0.5).
- `iou_threshold` - IOU above which two detections are considered duplicate and
  only the highest confidence one will be returned (default: 0.4).
- `slicing_threshold` - aspect ratio (short side over long side) below which
  the image will be cut into square slices as the model doesn't deal well with
  very non-square images (default: 0.7).
- `slice_overlap` - minimal ratio of the slice area that will be overlapped:
  overlaps are necessary to make sure the objects at slice boundaries get
  detected (default: 0.2).
