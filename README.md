# Wentral

A frontend for machine learning models that detect objects in web pages, that
can be used to:

- Measure and visualize the performance of object detectors on datasets,
- Expose an object detector as an HTTP web service.

## Installation

We recommend to install Wentral into a virtualenv. After the virtualenv is
activated execute `python setup.py install`. Dependencies will be installed
automatically.

## Web service

To make a web service from a `detector.Class` loading the weights from
`weights/file` run:

    $ wentral ws -d detector.Class -w weights/file

See [CLI docs](docs/cli.md#web-service) for more info on usage.

## Benchmarks

To benchmark `detector.Class` on a `data/set` run:

    $ wentral bm [-v] -d detector.Class -w weights/file data/set

See [CLI docs](docs/cli.md#benchmarking) for more info on usage.

## Development

Most common scenario will be implementing detectors to use with Wentral. The
[API docs](api.md) has [more detail](api.md#implementing-detectors) on this.

You are also welcome to contribute to Wentral itself. Make sure the tests still
pass and the coverage is not reduced. Make sure to follow
[eyeo coding style](https://adblockplus.org/coding-style#python) to make
reviews simpler.

### Testing

#### Python

We use [Tox][4] for testing Python code and Python linting. Install Tox with
pip if you haven't already and then run the tests with:

    $ tox

#### JavaScript

There's also a small amount of JavaScript in this repo. Unfortunately it has no
tests, but you can lint it using [ESLint][5] (more info on eyeo eslint config
[here][3]). Make sure you have ESLint and eyeo config installed:

    $ npm install -g eslint eslint-config-eyeo

and then run:

    $ eslint wentral/vis_ui/visualization.js

You only need to do it if you changed that file.

### CI

The CI setup in the GitLab repository runs both Python tests and JavaScript
linting. It's configured via [.gitlab-ci.yml](.gitlab-ci.yml).

## License

Wentral is Free and Open Source software distributed under the terms of MIT
license (see [LICENSE.txt](LICENSE.txt) for more details).


[3]: https://gitlab.com/eyeo/auxiliary/eyeo-coding-style/-/tree/master/eslint-config-eyeo
[4]: https://tox.readthedocs.io/en/latest/
[5]: https://eslint.org/
