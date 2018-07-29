#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts
time ./bodypart_detector_client_fpga.py --test-annotation-list ../config/test_annotation_list_old_new --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --detect-bodypart MouthHook,LeftDorsalOrgan,RightDorsalOrgan --verbosity 0  --crop-size 256 --dir-keypoints "${KEYPOINTS_DIR}/" --dir-descriptor "${DESCRIPTORS_DIR}/" --port-number 10000