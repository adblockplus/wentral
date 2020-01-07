# This file is part of Ad Detect YOLO <https://adblockplus.org/>,
# Copyright (C) 2019 eyeo GmbH
#
# Ad Detect YOLO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# Ad Detect YOLO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ad Detect YOLO. If not, see <http://www.gnu.org/licenses/>.

import os
from setuptools import setup

README_PATH = os.path.join(os.path.dirname(__file__), 'README.md')

with open(README_PATH) as f:
    long_description = f.read()

setup(
    name='ad-detect-yolo',
    version='0.1.0',
    description='Detect ads in screenshots using YOLO v.3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='eyeo GmbH',
    author_email='info@adblockplus.org',
    url='https://gitlab.com/eyeo/sandbox/ad-detect-yolo',
    packages=['ady'],
    entry_points={
        'console_scripts': [
            'adyws=ady.adyws:main',
            'adybm=ady.adybm:main',
        ],
    },
    include_package_data=True,
    install_requires=[
        'Flask',
        'numpy',
        'Paste',
        'Pillow',
        'psutil',
        'requests',
        'tensorflow==1.13.2',
        'waitress',
        'admincer',
    ],
    license='GPLv3',
    zip_safe=False,
    keywords='ad detection, yolo',
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
