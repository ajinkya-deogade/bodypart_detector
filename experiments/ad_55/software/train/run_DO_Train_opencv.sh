#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts

mkdir -vp ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/
./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_DO_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_1.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_1.p --training-bodypart RightDorsalOrgan
#./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_DO_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_2.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_2.p --training-bodypart RightDorsalOrgan
#./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_DO_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_1.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_1.p --training-bodypart LeftDorsalOrgan
#./ex_MHhook_detection_train_opencv.py --train-annotation-list ../../../config/forTraining/train_annotation_list_DO_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_2.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_2.p --training-bodypart LeftDorsalOrgan
