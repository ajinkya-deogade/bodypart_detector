#!/bin/bash
set -e
set -u
set -x

source ./sourceme

./compare_fpga_opencv_kp_orientation.py --train-annotation-list ../config/forTraining/train_annotation_list_DO_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --dir-keypoints "/Users/adeogade/Dropbox (CRG)/FPGA_Validation/TrainingData/validKeyPoints/" --dir-descriptor "/Users/adeogade/Dropbox (CRG)/FPGA_Validation/TrainingData/descriptors/"
