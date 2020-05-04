# This file is part of Ad Detect YOLO <https://adblockplus.org/>,
# Copyright (C) 2019-present eyeo GmbH
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

"""Common constants."""

# Detection parameters
CONF_THRESHOLD = 0.5     # Level of confidence that we count as detection.
IOU_THRESHOLD = 0.4      # IOU above which two boxes are considered the same.

# Slicing detector proxy defaults.
SLICING_THRESHOLD = 0.7
SLICE_OVERLAP = 0.2
