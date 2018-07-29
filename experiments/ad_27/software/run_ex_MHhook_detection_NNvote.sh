#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_DO_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_RightDO_1.p --negative-training-datafile ../expts/train_neg_RightDO_1.p --training-bodypart RightDorsalOrgan
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_DO_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_RightDO_2.p --negative-training-datafile ../expts/train_neg_RightDO_2.p --training-bodypart RightDorsalOrgan
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_DO_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_LeftDO_1.p --negative-training-datafile ../expts/train_neg_LeftDO_1.p --training-bodypart LeftDorsalOrgan
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_DO_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_LeftDO_2.p --negative-training-datafile ../expts/train_neg_LeftDO_2.p --training-bodypart LeftDorsalOrgan