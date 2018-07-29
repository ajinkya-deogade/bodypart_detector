#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

# ./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 1 --positive-training-datafile ../expts/train_pos_RightMHhook.p --negative-training-datafile ../expts/train_neg_RightMHhook.p --training-bodypart RightMHhook

./ex_MHhook_detection_test.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_RightMHhook.p --negative-training-datafile ../expts/train_neg_RightMHhook.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 1