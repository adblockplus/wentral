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

import os
from setuptools import setup

DESCRIPTION = 'Frontend for ML-based perceptual element detection on the Web'
README_PATH = os.path.join(os.path.dirname(__file__), 'README.md')

with open(README_PATH) as f:
    long_description = f.read()

setup(
    name='wentral',
    version='0.0.1',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='eyeo GmbH',
    author_email='info@adblockplus.org',
    url='https://gitlab.com/eyeo/machine-learning/wentral',
    packages=['wentral'],
    entry_points={
        'console_scripts': ['wentral=wentral.__main__:main'],
    },
    include_package_data=True,
    install_requires=[
        'Flask',
        'Paste',
        'Pillow',
        'psutil',
        'requests',
        'waitress',
        'admincer',
    ],
    license='MIT',
    zip_safe=False,
    keywords='object detection, web',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Utilities',
    ],
)
