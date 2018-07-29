#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./ex_MHhook_detection_test.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos.p --negative-training-datafile ../expts/train_neg.p --desc-dist-threshold 0.001 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --nthread 2