#! /bin/bash

set -e
set -u
set -x

source ./sourceme

#./train_detect_StratifiedShuffleSplit_allBodyPart_2.py --train-annotation-list-all ../config/temp --project-path "${PROJECT_PATH}/" --mh-neighborhood 256 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 200 --training-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --pos-neg-equal 0 --crop-size 256 --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}"

./train_detect_StratifiedShuffleSplit_allBodyPart_2.py --train-annotation-list-all ../config/annotation_list_old_new_all_forStratified --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 200 --training-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --pos-neg-equal 0 --crop-size 256 --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}"
./train_detect_StratifiedShuffleSplit_allBodyPart_2.py --train-annotation-list-all ../config/annotation_list_old_new_all_forStratified --project-path "${PROJECT_PATH}/" --mh-neighborhood 256 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 200 --training-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --pos-neg-equal 0 --crop-size 256 --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}"