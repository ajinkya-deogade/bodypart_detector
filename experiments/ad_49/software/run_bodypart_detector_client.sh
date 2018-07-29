#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#time ./bodypart_detector_client.py --test-annotation-list ../config/test/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart MouthHook --verbosity 1 --crop-size 512
time ./bodypart_detector_client.py --test-annotation-list ../config/test/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --detect-bodypart RightMHhook --verbosity 1 --crop-size 256
#time ./bodypart_detector_client.py --test-annotation-list ../config/test/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart LeftMHhook --verbosity 1 --crop-size 512
#time ./bodypart_detector_client.py --test-annotation-list ../config/test/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart RightDorsalOrgan --verbosity 1 --crop-size 512
#time ./bodypart_detector_client.py --test-annotation-list ../config/test/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart LeftDorsalOrgan --verbosity 1 --crop-size 512