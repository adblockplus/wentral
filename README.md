# Wentral

A frontend for ad detecting machine learning models. Provides benchmarking and
web service tools.

## Installation

Install into a virtualenv. After the virtualenv is activated execute
`python setup.py install`. Dependencies will be installed automatically.

If you would like to have stable dependency versions that are (more or less)
guaranteed to work, instead of the latests ones, you can run
`pip install -r requirements.txt` before running the command from previous
paragraph.

## Web service

With the virtualenv activated the web server can be started with:

    $ wentral ws -d MODEL_CLASS -w WEIGHTS_FILE [--port port]

It will listen on specified port (8080 by default), and you can interact with
it by pointing your browser to `http://localhost:8080/` or programmatically.
See `wws -h` for description of additional options.

Client code for Python is provided in `wentral.client`:

    from PIL import Image
    import wentral.client as cl

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

The defaults can be changed by supplying the options to `wentral ws` (see
`wentral ws -h` for more info).

There's also a GET endpoint for requesting server status at
`http://localhost:8080/status`. It returns a JSON document that contains the
information about server memory consumption and current active detection
requests (including their parameters).

### Using alternative ad detectors with `wentral ws`

Supply `--detector`/`-d` parameter with a fully qualified name of another
detector implementation (e.g. `some.other.AdDetector`) and provide
`--weights-file`/`-w` option, e.g.:

    $ wentral ws -d my.other.FancyAdDetector -w fancy.weights

## Benchmarks

`benchmark` (or `bm`) command measures ad detection performance. It can work
with the web service, model weights or a directory that contains marked
screenshots. You can also use detections from a previous run saved in a JSON
file (see below).

Wentral will load the images from `dataset_path`. It will also load the ground
truth from a `.csv` file in the same directory or from `.txt` files (in YOLOv3
format) that have the same names as the images.

Benchmark runs output overall statistics to standard output. You can request
numbers on individual images using `--verbose` or `-v` and/or more detailed
JSON output via `--output` or `-o`.

### Usage with a web service

Set `--detector` to `server` and provide `--server-url` option:

    $ wentral bm -d server -s http://localhost:8080/ dataset_path

This will upload image to `localhost:8080` expecting replies in the same format
as what `wentral ws` returns.

### Usage with YOLO or other object detection model and weights file

Set `--detector` to `yolo` or fully qualified name of another detector
implementation (e.g. `some.other.AdDetector`) and provide `--weights-file`
option:

    $ wentral bm -d yolo -w yolo_v3.weights dataset_path

This will load model weights from `yolo_v3.weights` and then use the resulting
model.

    $ wentral bm -d some.other.AdDetector -w other.weights dataset_path

This will import `AdDetector` from `some.other` module and instantiate it with
`weights_file="other.weights"` (the values of `--confidence-threshold` and
`--iou-threshold`) will also be passed to the constructor if it takes such
arguments.

### Usage with marked regions from another dataset

Set `--detector` to `static` and provide `--path` option:

    $ wentral bm -d static -p another_dataset dataset_path

This will take marked regions of `another_dataset` as detections and measure
this against `dataset_path` as the ground truth. It's important that the former
contain region marking for all images in the latter. Otherwise it's not
possible to produce detections for the missing images and to avoid giving out
misguiding results this is considered an error.

### Usage with JSON output of another benchmark run

Sometimes it could be useful to re-use the detections from another run. It can
be to try different values of confidence threshold or IoU threshold or to
produce visualizations. There's a built-in detector implementation that does
this:

    $ wentral bm -d json -p previous-run.json dataset_path

This will load detections from `previous-run.json` (that was produced using
`--output` option). Different confidence and IoU thresholds can be applied to
produce different results from the orginal run.

All detections with confidence above 0.001 will be included in the JSON output
(with duplicates removed), so it's possible to take an output of a run with
high confidence threshold and reevaluate it with a lower confidence threshold.
The reverse is also possible of course.

In addition to using JSON output as a source of detections, it can also be used
as the source of the ground truth. In order to do it, give the JSON file as the
dataset parameter:

    $ wentral bm -d yolo -w yolo.weights previous-run.json

Note: The JSON file contains the path to the original images, but it doesn't 
contain the images themselves. If the images are not where they were when the
JSON file was produced, the JSON file cannot be used as a dataset anymore.

### Visualizing model performance

If you add `--visualizations-path` (`-z`) option to benchmark run, an
additional visualization folder is generated. The folder will contain
`index.html`, some JavaScript and JSON and lots of images. The UI can be viewed
locally with `python -m http.server` or hosted at a more serious webserver, or
cloud static hosting like Amazon S3 or Google Cloud Storage. Then just point
your browser to `index.html`.

There are two sections in the visualization UI: screenshots and detections.

#### Screenshots

This section shows full screenshots with detection boxes drawn on top of them.
The colors of the displayed boxes will be as follows:

- Detection matching a ground truth box (true positive): green,
- Detection not matching a ground truth box (false positive): orange,
- Ground truth box matching a detection (also true positive): blue,
- Ground truth box not matching a detection (false negative): red.

All detections also have their corresponding confidence displayed on top of
them in %. You can adjust the size of screenshots using the slider at the top,
or zoom in with a mouse click on the screenshot.

It's possible to filter the screenshots to only see the ones that have false
positives or false negatives in them.

#### Detections

This section shows detections and ground truth boxes:

- Ground truth boxes that have been detected have a blue frame,
- Detections matching a ground truth box have a green frame,
- Detections not matching a ground truth box (false positive) have an orange
  frame,
- Undetected ground truth boxes (false negatives) have a red frame.

Similalry to the screenshots view, it's possible to zoom in by clicking on
detections. The zoom in view also shows the detection or ground truth box in
context of the screenshot that it comes from.

#### Similarity data

An additional file with detection/ground truth similarity information can be
placed into the same directory named `nn.json`. It should contain a mapping of
fragment file names to arrays of mappings with two keys: `file` and `dist` that
contain file names and distances of similar fragments, for example:

    {
      "screenshot_42_868,1049-1171,1258.png": [
        {
          "file": "screenshot_1337_422,527-1176,616.png",
          "dist": 0.586823433637619
        },
        {
          "file": "screenshot_18_970,1098-1276,1363.png",
          "dist": 0.5552344620227814
        },
      ]
    }

When similarity information is provided, similar images are displayed in the
zoom in view below the full screenshot (for those fragments for which
similarity is provided).

## Testing

### Python

We use [Tox][4] for testing Python code and Python linting. Install Tox with
pip if you haven't already and then run the tests with:

    $ tox

### JavaScript

There's also a tiny bit of JavaScript in this repo. Unfortunately it has no
tests, but you can lint it using [ESLint][5] (more info on eyeo eslint config
[here][3]). Make sure you have ESLint and eyeo config installed:

    $ npm install -g eslint eslint-config-eyeo

and then run:

    $ eslint wentral/vis_ui/visualization.js

You only need to do it if you changed that file.

### CI

The CI setup in the GitLab repository runs both Python tests and JavaScript
linting. It's configured via [.gitlab-ci.yml](.gitlab-ci.yml).

[1]: https://github.com/ftramer/ad-versarial/releases
[2]: https://github.com/ftramer/ad-versarial/
[3]: https://gitlab.com/eyeo/auxiliary/eyeo-coding-style/-/tree/master/eslint-config-eyeo
[4]: https://tox.readthedocs.io/en/latest/
[5]: https://eslint.org/
