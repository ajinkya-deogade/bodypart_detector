#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts

#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart MouthHook --verbosity 1 --crop-size 256 --dir-keypoints "F:/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/FPGA/test/keypoints/" --port-number 9998
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart LeftMHhook --verbosity 1 --crop-size 256 --dir-keypoints "F:/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/FPGA/test/keypoints/" --port-number 9999
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart RightMHhook --verbosity 1 --crop-size 256 --dir-keypoints "F:/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/FPGA/test/keypoints/" --port-number 10000
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart LeftDorsalOrgan --verbosity 1 --crop-size 256 --dir-keypoints "F:/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/FPGA/test/keypoints/" --port-number 10001
#time ./bodypart_detector_client_fpga.py --test-annotation-list ../../../config/forTesting/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --test-bodypart RightDorsalOrgan --verbosity 1 --crop-size 256 --dir-keypoints "F:/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/FPGA/test/keypoints/" --port-number 10002
