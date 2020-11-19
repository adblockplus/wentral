# Wentral file formats

Wentral creates and consumes several file formats that are described below.

## Dataset regions

Datasets generally come as a directory of images that contain an additional
file or files with region markings. Region markings come in CSV or TXT formats.

### CSV

CSV region markings are a single file that contains the following columns:

- `image` -- file name of the image to which the region belongs.
- `xmin`, `ymin`, `xmax`, `ymax` -- bounding box of the region; the coordinates
  are from 0 to width / height - 1.
- `label` -- a text string, indicating the type for this region.

### TXT (YOLO format)

TXT region markings are one file per image, that has the same name as the image
but with `.txt` extension (instead of `.png` or `.jpg`). The file contains one
line per region, that has the following format:

   TYPE X Y WIDTH HEIGHT

Here `TYPE` is a non-negative integer that indicates the region type, `X` and
`Y` are floats between 0 and 1 that are coordinates of the center of the region
and `WIDTH` and `HEIGHT` are floats between 0 and 1 that are width and height
of the region. All sizes and coordinates are relative to the image size.

TXT region files normally come accompanied by `region.names` file that contains
the names of the region types: the first line corresponds to `TYPE` 0, second
line to `TYPE` 1 and so forth.

## JSON output of benchmarks

Benchmarking detectors with `wentral bm ... -o output.json` produces a detailed
results file in JSON format. The content of the file is an object with the
following keys:

- `dataset` -- Dataset description.
- `detector` -- Detector and its parameters.
- `image_count` -- Number of images in the dataset.
- `images_path` -- Path to the dataset directory.
- `tp` -- True positives count.
- `fp` -- False positives count.
- `fn` -- False negatives count.
- `precision`, `recall`, `f1` -- Precision, recall and F1 score.
- `mAP` -- Mean average precision (it's actually simple average precision
  because Wentral only has one class or regions).
- `images` -- Array of objects that contain information about individual
  images. Each object contains the following keys:
  - `image_name` -- File name of the image.
  - `confidence_threshold` -- Confidence threshold that's used to filter the
    detections before they are compared to the ground truth.
  - `match_iou` -- IoU necessary for a detection to match a ground truth box.
  - `tp`, `fp`, `fn`, `precision`, `recall`, `f1` -- Results for this one
    image.
  - `detections` -- Array of arrays that contain coordinates of detections
    (X0, Y0, X1, Y1) followed by detection confidence and `true`/`false`
    depending on whether this detection matches a ground truth box. Note that
    only detections with confidence over the threshold are counted for
    evaulating model performance.
  - `ground_truth` -- Array of arrays that contain coordinates of ground truth
    boxes (X0, Y0, X1, Y1) followed by detection confidence. If there are no
    matching detections, detection confidence will be 0.

## Output of benchmark visualization

Running `wentral bm ... -z vis-dir` produces a directory with visualization UI.
It consists of the original images with detection boxes drawn on them, smaller
images of detections and ground truth boxes, `index.html` that contains the
HTML of the UI, `visualization.js` that contains the logic and `data.js` that
contains the data. `data.json` contains a list of objects with the following
keys:

- `name` -- File name of the image.
- `tp`, `fp`, `fn` -- Number of true / false positives and false negatives.
- `detections` -- Object with two keys: `true` and `false`, that each contain
  an array of object with the following keys:
  - `file` -- File name of the detection image.
  - `box` -- Array that contains the box coordinates and detection confidence.
- `ground_truth` -- Object with two keys: `detected` and `missed`, that each
  contain an array of objects with the following keys:
  - `file` -- File name of the detection image.
  - `box` -- Array that contains the box coordinates and detection confidence
    (detection confidence might be 0).

### Detection similarity data

Visualization UI can also display ground truth boxes similar to false positives
and false negatives. For this a detection similarity file should be placed in
the visualization directory. It should be named `nn.json` and contain an object
that maps file names of the detections / ground truth images to an array of
objects with the following keys:

- `file` -- File name of the similar image.
- `dist` -- Distance between the original image and the similar one.
