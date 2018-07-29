#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#time ./bodypart_detector_client.py --test-annotation-list ../config/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 0 --save-dir-keypoints /Users/adeogade/work/mhdo/cropped_images/20150312_keypoints_nOctave_2_OctaveLayers_3/
time ./bodypart_detector_client.py --test-annotation-list ../config/train_list_20141220 --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-keypoints "/Users/adeogade/Dropbox (CRG)/FPGA_Validation/OpenCV_KP_DESC/keypoints/" --save-dir-images "/Users/adeogade/Dropbox (CRG)/FPGA_Validation/OpenCV_KP_DESC/images/" --save-dir-descriptors "/Users/adeogade/Dropbox (CRG)/FPGA_Validation/OpenCV_KP_DESC/descriptors/"