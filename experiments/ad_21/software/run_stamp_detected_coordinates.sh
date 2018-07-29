#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./stamp_detected_coordinates.py --estimated-coordinate-file ../expts/estimated_coordinate_file.txt --input-coordinate-file ../expts/input_coordinate_file.txt --display 0 --video-file-one ../expts/one.avi --video-file-two ../expts/two.avi --save-video-file-one ../expts/one_stamp.avi --save-video-file-two ../expts/two_stamp.avi


# --project-path "${PROJECT_PATH}/"



