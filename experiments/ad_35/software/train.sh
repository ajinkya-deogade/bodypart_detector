#!/bin/bash
set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#./ex_MHhook_detection_train.py --train-annotation-list ../config/train_list_20141127 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_20141216.p --negative-training-datafile ../expts/train_neg_20141216.p --training-bodypart MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan
./ex_MHhook_detection_train.py --train-annotation-list ../config/train_list_20141220 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --positive-training-datafile ../expts/train_pos_20141220.p --negative-training-datafile ../expts/train_neg_20141220.p --training-bodypart MouthHook,RightDorsalOrgan,LeftDorsalOrgan