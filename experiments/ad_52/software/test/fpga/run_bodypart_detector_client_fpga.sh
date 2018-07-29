#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts

time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart MouthHook --verbosity 1 --crop-size 256 --dir-keypoints "${KEYPOINTS_DIR}/" --dir-descriptor "${DESCRIPTORS_DIR}/" --port-number 10000
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart LeftMHhook --verbosity 1 --crop-size 256 --dir-keypoints "${KEYPOINTS_DIR}/" --dir-descriptor "${DESCRIPTORS_DIR}/" --port-number 11000
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart RightMHhook --verbosity 1 --crop-size 256 --dir-keypoints "${KEYPOINTS_DIR}/" --dir-descriptor "${DESCRIPTORS_DIR}/" --port-number 12000
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart LeftDorsalOrgan --verbosity 1 --crop-size 256 --dir-keypoints "${KEYPOINTS_DIR}/" --dir-descriptor "${DESCRIPTORS_DIR}/" --port-number 13000
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart RightDorsalOrgan --verbosity 1 --crop-size 256 --dir-keypoints "${KEYPOINTS_DIR}/" --dir-descriptor "${DESCRIPTORS_DIR}/" --port-number 14000