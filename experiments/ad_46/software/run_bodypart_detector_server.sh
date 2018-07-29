#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#./bodypart_detector_server.py --positive-training-datafile ../expts/20150518_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../expts/20150518_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1

#./bodypart_detector_server.py --positive-training-datafile ../expts/20150326_Combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../expts/20150326_Combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1

./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../expts/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1