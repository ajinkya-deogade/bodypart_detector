#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#for socket_port in {9998..9998}
#do
    ./bodypart_detector_server.py --positive-training-datafile ../expts/fpga/20160120_Hessian_250_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_all.p --negative-training-datafile ../expts/fpga/20160120_Hessian_250_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_all.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --detect-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --verbosity 1 --socket-port 9998
#    ./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_20141216.p --negative-training-datafile ../expts/train_neg_20141216.p --desc-dist-threshold 0.15 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --socket-port "${socket_port}" --detect-bodypart LeftDorsalOrgan,RightDorsalOrgan --verbosity 1 &
#done
