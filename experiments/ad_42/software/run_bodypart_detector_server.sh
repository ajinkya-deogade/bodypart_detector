#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#for socket_port in `seq 9998 10005`
#do
./bodypart_detector_server.py --positive-training-datafile ../expts/20150326_Combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../expts/20150326_Combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --socket-port 9998
#done


