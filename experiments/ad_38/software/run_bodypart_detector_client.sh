#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

time ./bodypart_detector_client.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 0 --dir-keypoints time ./bodypart_detector_client.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 0 --dir-keypoints "/Users/adeogade/Documents/MATLAB/HighResTracker/data/20150324_FPGA_keypoints/" --dir-images "/Users/adeogade/Documents/MATLAB/HighResTracker/data/cropped_images/"