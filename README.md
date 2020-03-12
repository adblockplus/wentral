# Ad Detect YOLO

This repository contains a web service that detects ads on screenshots using
YOLO v.3 object detection model.

## Installation

Install into a virtualenv. After the virtualenv is activated execute
`python setup.py install`. Dependencies will be installed automatically.

If you would like to have stable dependency versions that are (more or less)
guaranteed to work, instead of the latests ones, you can run
`pip install -r requirements.txt` before running the command from previous
paragraph.

## Configuration

This repository doesn't contain the weights file for the model. It can be
obtained from [releases page][1] of Ad-Versarial [repository][2] (download
`models.tar.gz` and take `page_based_yolov3.weights` from it. You can also
train from scratch and save your own weights (recommended).

## Running the web service

With the virtualenv activated the web server can be started with:

    $ adyws -w weights_file [--port port]

It will listen on specified port (8080 by default), and you can interact with
it by pointing your browser to `http://localhost:8080/` or programmatically.
See `adyws -h` for description of additional options.

Instead of providing `-w` option, you can set `YOLOv3_WEIGHTS_PATH` environment
variable to the path of the weights file. **This is supported for backward
compatibility but is deprecated and at some point `-w` will become mandatory**.

Client code for Python is provided in `ady.client`:

    from PIL import Image
    import ady.client as cl

    detector = cl.ProxyAdDetector('http://localhost:8080/')
    path = 'path/image.png'
    image = Image.open(path)
    ad_boxes = detector.detect(image, path)

When using other languages upload the image to `http://localhost:8080/detect`
using a POST request with Content-Type `multipart/form-data`. The field name
should be called `image`. The result is returned as a JSON document.

The requests to `detect` endpoint can also include additional parameters for
the detection process:

- `confidence_threshold` - minimum model confidence for detections to be
  returned (default: 0.5).
- `iou_threshold` - IOU above which two detections are considered duplicate and
  only the highest confidence one will be returned (default: 0.4).
- `slicing_threshold` - aspect ratio (short side over long side) below which
  the image will be cut into square slices as the model doesn't deal well with
  very non-square images (default: 0.7).
- `slice_overlap` - minimal ratio of the slice area that will be overlapped:
  overlaps are necessary to make sure the ads at slice boundaries get detected
  (default: 0.2).

The defaults can be changed by supplying the options to `adyws` (see `adyws -h`
for more info).

There's also a GET endpoint for requesting server status at
`http://localhost:8080/status`. It returns a JSON document that contains the
information about server memory consumption and current active detection
requests (including their parameters).

### Using alternative ad detectors with `adyws`

Supply `--detector`/`-d` parameter with a fully qualified name of another
detector implementation (e.g. `some.other.AdDetector`) and provide
`--weights-file`/`-w` option, e.g.:

    $ adyws -d my.other.FancyAdDetector -w fancy.weights

## Running the measurement script

There's also a script that measures ad detection performance. It can work with
a web service, model weights or a directory that contains marked screenshots.

The script will load the images from `dataset_path`. It will also load the
ground truth from a `.csv` file in the same directory or from `.txt` files (in
YOLOv3 format) that have the same names as the images.

The script outputs overall statistics to standard output. You can request
numbers on individual images using `--verbose` or `-v` and/or more detailed
JSON output via `--output` or `-o`.

### Usage with a web service

Set `--detector` to `server` and provide `--server-url` option:

    $ adybm -d server -s http://localhost:8080/ dataset_path

This will upload image to `localhost:8080` expecting replies in the same format
as what `adyws` returns.

### Usage with YOLO or other object detection model and weights file

Set `--detector` to `yolo` or fully qualified name of another detector
implementation (e.g. `some.other.AdDetector`) and provide `--weights-file`
option:

    $ adybm -d yolo -w yolo_v3.weights dataset_path

This will load model weights from `yolo_v3.weights` and then use the resulting
model.

    $ adybm -d some.other.AdDetector -w other.weights dataset_path

This will import `AdDetector` from `some.other` module and instantiate it with
`weights_file="other.weights"` (the values of `--confidence-threshold` and
`--iou-threshold`) will also be passed to the constructor if it takes such
arguments.

### Usage with marked regions from another dataset

Set `--detector` to `static` and provide `--path` option:

    $ adybm -d static -p another_dataset dataset_path

This will take marked regions of `another_dataset` as detections and measure
this against `dataset_path` as the ground truth. It's important that the former
contain region marking for all images in the latter. Otherwise it's not
possible to produce detections for the missing images and to avoid giving out
misguiding results this is considered an error.

### Visualizing model performance

You can produce visualizations with detection boxes and ground truth displayed
on top of the original images. Use `--visualizations-path` (`-z`) option
followed by the output directory for the visualizations.

The colors of the displayed boxes will be as follows:

- Detection matching a ground truth box (true positive): green,
- Detection not matching a ground truth box (false positive): orange,
- Ground truth box matching a detection (also true positive): blue,
- Ground truth box not matching a detection (false negative): red.

All detections also have their corresponding confidence displayed on top of
them in %.

## Testing

Make sure that `YOLOv3_WEIGHTS_PATH` is set and run the tests with `tox`.
Currently there's just one test and it expects the weights to be from the
Ad-versarial repo.

[1]: https://github.com/ftramer/ad-versarial/releases
[2]: https://github.com/ftramer/ad-versarial/
