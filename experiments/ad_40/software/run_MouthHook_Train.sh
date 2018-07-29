#!/bin/bash
set -e
set -u
set -x

source ./sourceme

./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MH_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_1.p --negative-training-datafile ../expts/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_1.p --training-bodypart MouthHook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MH_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_2.p --negative-training-datafile ../expts/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_2.p --training-bodypart MouthHook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MH_3 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_3.p --negative-training-datafile ../expts/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_3.p --training-bodypart MouthHook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MH_4 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_4.p --negative-training-datafile ../expts/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_4.p --training-bodypart MouthHook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MH_5 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_5.p --negative-training-datafile ../expts/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_5.p --training-bodypart MouthHook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MH_6 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_6.p --negative-training-datafile ../expts/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_6.p --training-bodypart MouthHook