#!/bin/sh
#
# This script installs pinned dependencies and checks that setup.py doesn't
# override them.

pip install -r requirements.txt
pip install . | tee pinned-install.log

if cat pinned-install.log | grep Uninstalling; then
    echo "Uninstalls detected: requirements.txt is not aligned with setup.py"
    exit 1
else
    echo "No uninstalls detected. All good."
fi
