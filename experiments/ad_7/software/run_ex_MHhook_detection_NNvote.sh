#!/bin/bash

set +e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#./ex_MHhook_detection_train.py --train-annotation-list ../config/train_annotation_list --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 1 --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --training-bodypart RightDO
#./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/1_20140214R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
#./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/5_20140213R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
#./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/10_20140214R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_LeftDO.p --negative-training-datafile ../expts/train_neg_LeftDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/15_20140213R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_LeftDO.p --negative-training-datafile ../expts/train_neg_LeftDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/16_20140214R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_LeftDO.p --negative-training-datafile ../expts/train_neg_LeftDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/5_20140213R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_LeftDO.p --negative-training-datafile ../expts/train_neg_LeftDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/11_20140214R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/7_20140213R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
./ex_MHhook_detection_test.py --video-file /f/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/12_20140214R.mp4 --project-path "${PROJECT_PATH}/" --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --save-dir-images /f/Detected/ --display 0
