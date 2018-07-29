#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../expts

./bodypart_detector_server.py --positive-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook.p --negative-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1