#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_MouthHook.p --negative-training-datafile ../expts/train_neg_MouthHook.p --training-bodypart MouthHook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_RightMHhook.p --negative-training-datafile ../expts/train_neg_RightMHhook.p --training-bodypart RightMHhook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_LeftMHhook.p --negative-training-datafile ../expts/train_neg_LeftMHhook.p --training-bodypart LeftMHhook