#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

time ./bodypart_detector_client.py --annotation-list ../config/forTesting/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-keypoints "C:/Users/deogadea/Dropbox (CRG)/FPGA_Validation/TestData/keypoints_OpenCV_NoOrientation/"

time ./bodypart_detector_client.py --annotation-list ../config/forTraining/train_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-keypoints "C:/Users/deogadea/Dropbox (CRG)/FPGA_Validation/TrainingData/keypoints_OpenCV_NoOrientation/"
