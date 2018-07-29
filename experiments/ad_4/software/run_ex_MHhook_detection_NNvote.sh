#!/bin/bash

set -e
set -u
set -x

mkdir -vp ../expts

./ex_MHhook_detection_train.py --train-annotation-list //Users/agomez/work/dev/mhdo/experiments/ad_4/config/train_annotation_list --project-path /Volumes/HD2/MHDO_Tracking/ --mh-neighborhood 50 --display 1 --training-datafile ../expts/train_MouthHook_20140527.p --training-bodypart MouthHook
./ex_MHhook_detection_test.py --test-annotation-list //Users/agomez/work/dev/mhdo/experiments/ad_4/config/test_annotation_list --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/train_MouthHook_20140527.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images_MouthHook_20140527/ --save-dir-error /Users/agomez/work/data/saved_error_MouthHook_20140527/
