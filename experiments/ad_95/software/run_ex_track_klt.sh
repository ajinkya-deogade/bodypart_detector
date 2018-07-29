#!/bin/bash

set -e
set -u
set -x

set +u
export PYTHONPATH=~/work/ext/opencv-2.4.8/samples/python2:"${PYTHONPATH}"
set -u

./track_klt.py /tmp/foo.mp4
