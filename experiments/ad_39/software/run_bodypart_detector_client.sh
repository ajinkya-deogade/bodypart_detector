#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./bodypart_detector_client.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1  --test-bodypart MouthHook --dir-keypoints "/Users/adeogade/Documents/MATLAB/HighResTracker/data/20150324_FPGA_keypoints/"