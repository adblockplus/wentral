# This file is part of Wentral
# <https://gitlab.com/eyeo/machine-learning/wentral/>,
# Copyright (C) 2019-present eyeo GmbH
#
# Wentral is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# Wentral is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Wentral. If not, see <http://www.gnu.org/licenses/>.


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
    url='https://gitlab.com/eyeo/sandbox/ad-detect-yolo',
    packages=['wentral'],
    entry_points={
        'console_scripts': [
            'wws=wentral.wws:main',
            'wbm=wentral.wbm:main',
        ],
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
    license='GPLv3',
    zip_safe=False,
    keywords='object detection, web',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Utilities',
    ],
)
