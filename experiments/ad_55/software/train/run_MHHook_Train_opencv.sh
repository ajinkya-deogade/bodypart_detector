#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3
./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MHhooks_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_1.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_1.p --training-bodypart RightMHhook
./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MHhooks_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_2.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_2.p --training-bodypart RightMHhook
./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MHhooks_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_1.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_1.p --training-bodypart LeftMHhook
./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MHhooks_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_2.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_2.p --training-bodypart LeftMHhook
