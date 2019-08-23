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

## Running the web server

With the virtualenv activated the web server can be started with: `adyws`.

## Testing

Make sure that `YOLOv3_WEIGHTS_PATH` is set and run the tests with `tox`.
Currently there's just one test and it expects the weights to be from the
Ad-versarial repo.

[1]: https://github.com/ftramer/ad-versarial/releases
[2]: https://github.com/ftramer/ad-versarial/
