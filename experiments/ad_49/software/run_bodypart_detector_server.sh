#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#./bodypart_detector_server.py --positive-training-datafile ../expts/20150409_Hessian_500_nOctaves_3_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_3_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../expts/20150409_Hessian_500_nOctaves_3_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --verbosity 0
./bodypart_detector_server.py --positive-training-datafile ../expts/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../expts/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --verbosity 0