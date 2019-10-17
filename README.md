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
`models.tar.gz` and take `page_based_yolov3.weights` from it. Set the
environment variable `YOLOv3_WEIGHTS_PATH` to point to the weights file.

## Running the web service

With the virtualenv activated the web server can be started with: `adyws`. It
will listen on port 8080, and you can interact with it by pointing your browser
to `http://localhost:8080/` or programmatically.

Client code for Python is provided in `ady.client`:

    from PIL import Image
    import ady.client as cl

    detector = cl.ProxyAdDetector('http://localhost:8080/')
    path = 'path/image.png'
    image = Image.open(path)
    ad_boxes = detector.detect(image, path)

When using other languages upload the image to `http://localhost:8080/detect`
using a POST request with Content-Type `multipart/form-data`. The field name
should be called `image`.

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

Provide `--server-url` or `-s` option:

    $ adybm -s http://localhost:8080/ dataset_path

This will upload image to `localhost:8080` expecting replies in the same format
as what `adyws` returns.

### Usage with a weights file

Provide `--weights-file` or `-w` option:

    $ adybm -w yolo_v3.weights dataset_path

This will load model weights from `yolo_v3.weights` and then use the resulting
model.

### Usage with another dataset

Provide `--marked-regions` or `-r` option:

    $ adybm -r another_dataset dataset_path

This will take marked regions of `another_dataset` as detections and measure
this against `dataset_path` as the ground truth. It's important that the former
contain region marking for all images in the latter. Otherwise it's not
possible to produce detections for the missing images and to avoid giving out
misguiding results this is considered an error.

Note: the measurement script currently depends on [bimed][3], which is not
listed as a dependency because it's not published on PyPI. You will need to
install it manually:

    $ pip install git+https://gitlab.com/eyeo/machine-learning/bimed

## Testing

Make sure that `YOLOv3_WEIGHTS_PATH` is set and run the tests with `tox`.
Currently there's just one test and it expects the weights to be from the
Ad-versarial repo.

Note: because of dependency on [bimed][3] (that is not public yet) the tests
require a bimed checkout to be located at `../bimed`.

[1]: https://github.com/ftramer/ad-versarial/releases
[2]: https://github.com/ftramer/ad-versarial/
[3]: https://gitlab.com/eyeo/sandbox/bimed
