#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./ex_MHhook_detection_test.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/20160319_OpenCV_Training/MouthHook_positive.p --negative-training-datafile ../expts/20160319_OpenCV_Training/MouthHook_negative.p --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --outlier-error-dist 10 --display 0 --nthread 2 --detect-bodypart "MouthHook"
#./ex_MHhook_detection_test.py --test-annotation-list ../config/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/20160319_OpenCV_Training/LeftMHhook_positive.p --negative-training-datafile ../expts/20160319_OpenCV_Training/LeftMHhook_negative.p --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --outlier-error-dist 10 --display 0 --nthread 2
#./ex_MHhook_detection_test.py --test-annotation-list ../config/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/20160319_OpenCV_Training/RightMHhook_positive.p --negative-training-datafile ../expts/20160319_OpenCV_Training/RightMHhook_negative.p --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --outlier-error-dist 10 --display 0 --nthread 2
#./ex_MHhook_detection_test.py --test-annotation-list ../config/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/20160319_OpenCV_Training/LeftDorsalOrgan_positive.p --negative-training-datafile ../expts/20160319_OpenCV_Training/LeftDorsalOrgan_negative.p --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --outlier-error-dist 10 --display 0 --nthread 2
#./ex_MHhook_detection_test.py --test-annotation-list ../config/test_annotation_list_fpgaKNNVal --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/20160319_OpenCV_Training/RightDorsalOrgan_positive.p --negative-training-datafile ../expts/20160319_OpenCV_Training/RightDorsalOrgan_negative.p --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --outlier-error-dist 10 --display 0 --nthread 2