#!/bin/bash

set -e
set -u
set -x

source ./sourceme

./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MHhooks_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_RightMHhook_1.p --negative-training-datafile ../expts/train_neg_RightMHhook_1.p --training-bodypart RightMHhook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MHhooks_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_RightMHhook_2.p --negative-training-datafile ../expts/train_neg_RightMHhook_2.p --training-bodypart RightMHhook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MHhooks_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_LeftMHhook_1.p --negative-training-datafile ../expts/train_neg_LeftMHhook_1.p --training-bodypart LeftMHhook
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list_MHhooks_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_LeftMHhook_2.p --negative-training-datafile ../expts/train_neg_LeftMHhook_2.p --training-bodypart LeftMHhook