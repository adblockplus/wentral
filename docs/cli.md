# Wentral command line usage

Wentral contains a command line wrapper (called `wentral`) that allows access
to most features. When installing Wentral with `pip install` it will be placed
into the `bin/` directory where Python and other script wrappers live. The
CLI has two subcommands that are described below.

## Web service

Wentral can expose any [detector implementation](api.md#implementing-detectors)
as a [web service](api.md#using-the-web-service) that can be used to perform
multiple benchmarking runs without reloading the model or as part of production
setup.

The basic invocation goes like this:

    $ wentral ws -d DETECTOR_CLASS -w WEIGHTS_FILE [--port PORT]

The web service will listen on specified `PORT` (8080 by default), and you can
interact with it by pointing your browser to `http://localhost:8080/` or
[programmatically](api.md#using-the-web-service). See `wentral ws -h` for
description of additional options or see
[detector arguments description](#detector-arguments) below.

There's also a GET endpoint for requesting server status at
`http://localhost:8080/status`. It returns a JSON document that contains the
information about server memory consumption and current active detection
requests (including their parameters).

### Slicing proxy

What `wentral ws` exposes is actually not the detector class itself. Instead it
wraps it with `SlicindDetectorProxy` from `wentral.slicing_detector_proxy`. The
proxy can deal with screenshots that are very far from square share by slicing
them into square fragments, running detection on fragments (it calls
`batch_detect` method of the original detector class) and then combining the
detections from the fragments into detections from the whole screenshot.

To configure the slicing, `wentral ws` takes the following additional
parameters:

- `--slicing-threshold` -- If the aspect ratio (short axis, typically width,
  divided by long axis, typically height) of the screenshot is equal or less
  than this number, it will be sliced. Otherwise slicing proxy just passes it
  through to the wrapped detector. The default value for this is 0.7.
- `--slice-overlap` -- When non-square images are cut into slices, the slices
  will overlap each other by some percentage of their area. This parameter
  configures the overlap percentage. Default value is 0.2.

## Benchmarking

Wentral can also be used to measure the performance of
[detector implementations](api.md#implementing-detectors):

    $ wentral bm -d DETECTOR_CLASS DATASET_PATH

You will need to supply detector-specific parameters like `--weights-file`/`-w`
to initialize the detector class. Run `wentral bm -h` to get more help on
those or see [detector arguments description](#detector-arguments) below.

Wentral will load the images from `DATASET_PATH`. It will also load the ground
truth from a CSV file in the same directory or from TXT files (in YOLOv3)
format) that have the same names as the images.

Benchmark runs output overall statistics to standard output. You can request
numbers on individual images using `--verbose`/`-v` and/or more detailed JSON
output via `--output` or `-o`.

Other command line arguments that affect benchmarking are:

- `--confidence-threshold`/`-c` -- Only detections with confidence equal or
  greater than this are counted for the purposes of calculating true and false
  positives, false negatives, precision, recall and f1 score. For mAP all
  detections are used, they are also all stored in JSON output if it's
  requested. The default confidence threshold is 0.5.
- `--match-iou`/`-m` -- Only detections that have sufficient overlap with the
  ground truth boxes are counted as true positives. Overlap is measured by
  intersection over union (IoU) and this argument sets IoU value at which
  detections are considered to be matching the ground truth. The default value
  for this argument is 0.4.
- `--visualizations-path`/`-z` -- Create a [visualization](#visualization) of
  detections and ground truth boxes and save it in the directory specified by
  this parameter.

## Visualization

When visualization is requested, `wentral bm` will create an additional
visualization folder. The folder will contain `index.html`, some JavaScript and
JSON and some images. The UI can be viewed locally with `python -m
http.server` or hosted at a more serious webserver, or cloud static hosting
like Amazon S3 or Google Cloud Storage. Then just point your browser to
`index.html`.

There are two sections in the visualization UI: screenshots and detections.

### Screenshots

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

### Detections

This section shows detections and ground truth boxes:

- Ground truth boxes that have been detected have a blue frame,
- Detections matching a ground truth box have a green frame,
- Detections not matching a ground truth box (false positive) have an orange
  frame,
- Undetected ground truth boxes (false negatives) have a red frame.

Similarly to the screenshots view, it's possible to zoom in by clicking on
detections. The zoom in view also shows the detection or ground truth box in
context of the screenshot that it comes from.

### Similarity data

An additional file with detection/ground truth [similarity
information](file-formats.md#detection-similarity-data) can be placed into the
same directory named `nn.json`. When similarity information is provided,
similar images are displayed in the zoom in view below the full screenshot (for
those fragments for which similarity is provided).

## Detector arguments

Most detectors are Python classes that wrap machine learning models, usually
neural networks. The arguments that are normally provided to them are:

- `--weights-file`/`-w` -- path of the file that contains model weights.
- `--iou-threshold` -- Usually the models discard detections that have too much
  overlap with other detections. Overlap is measured by intersection over union
  (IoU) and this argument sets IoU value at which detections are considered to
  be duplicates. The default value for this is 0.4.

In addition, you can use `--extra`/`-x` (the parameter takes the form
`ARGNAME=VALUE`) to pass additional parameters to the constructor of the
detector class.

### Special detectors

In addition to ML-based detectors, there are 3 built-in special detectors that
can be specified as the value of `-d` argument:

- `json` -- This detector needs `--path`/`-p` argument pointing to a JSON file
  that was earlier produced by `wentral bm ... -o JSON_FILE`. It will load
  detections from this JSON file and it allows recalculating the results with
  different confidence threshold and match IoU values.
- `server` -- This detector needs `--server-url`/`-s` argument with URL of a
  server that runs `wentral ws`. Mostly useful for running several benchmarks
  on the same model without reloading the weights.
- `static` -- This detector needs `--path`/`-p` argument that points to a
  directory containing a dataset (in the same format as `DATASET`). The
  detector returns marked regions in that dataset as detections. Of course if
  the two datasets must contain the same images. This is useful for evaluating
  datasets labeled by humans to establish a human level baseline.
