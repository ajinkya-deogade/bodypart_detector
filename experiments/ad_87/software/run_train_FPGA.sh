#! /bin/bash

set -e
set -u
set -x

source ./sourceme

#./train_FPGA_corrected.py --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 200 --training-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --pos-neg-equal 0 --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 10 --crop-size 256 --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}"
#./train_FPGA_corrected.py --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 200 --training-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --pos-neg-equal 0 --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 10 --crop-size 256 --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}"
./train_FPGA_corrected_final.py --project-path "${PROJECT_PATH}/" --mh-neighborhood 512 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 200 --training-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --pos-neg-equal 0 --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 10 --crop-size 256 --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}"