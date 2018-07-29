#! /bin/bash

set -e
set -u
set -x

source ./sourceme

## 7c2 - backlight
# python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20180417_re --test-annotation-list-all ../config/dataCollectedOn_20180417_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 5

## 7c2 - top light
# python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20170317_re --test-annotation-list-all ../config/dataCollectedOn_20170317_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 5

## 7c0 - back light
# python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20180417_re --test-annotation-list-all ../config/dataCollectedOn_20180417_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 7

## 7c0 - top light
# python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20170317_re --test-annotation-list-all ../config/dataCollectedOn_20170317_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 7

## 7c1 - back light
#python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20180417_re --test-annotation-list-all ../config/dataCollectedOn_20180417_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 6

## 7c1 - top light
#python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20170317_re --test-annotation-list-all ../config/dataCollectedOn_20170317_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 6

## 7c1 - top light - new train added
#python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20180417_re2 --test-annotation-list-all ../config/dataCollectedOn_20180417_re2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 6

## 7c1 - top light - new train added
python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20180417_re2 --test-annotation-list-all ../config/dataCollectedOn_20180417_re2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 100 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 6

## 7c1 - back light - 25 pixels neighbourhood
#python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20180417_re --test-annotation-list-all ../config/dataCollectedOn_20180417_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 25 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 6

## 7c0 - top light
#python train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/dataCollectedOn_20170318_re --test-annotation-list-all ../config/dataCollectedOn_20170318_re --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 100 --training-bodypart "MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan,CenterBolwigOrgan,LeftBolwigOrgan,RightBolwigOrgan" --pos-neg-equal 0 --desc-dist-threshold 0.0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 15 --crop-size 256  --fpga-dir-kp "${KEYPOINTS_DIR}/" --fpga-dir-desc "${DESCRIPTORS_DIR}" --num-train 5