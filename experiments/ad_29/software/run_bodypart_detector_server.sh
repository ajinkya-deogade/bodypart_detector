#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

for socket_port in 9994
do
    ./bodypart_detector_server.py --positive-training-list C:/Users/deogadea/Documents/mhdo/experiments/ad_29/expts/train_neg_Hessian_600_nOctaves_3_nOctaveLayers_3_LeftMHhook_all.p --negative-training-list C:/Users/deogadea/Documents/mhdo/experiments/ad_29/expts/train_neg_Hessian_600_nOctaves_3_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --verbosity 1 --socket-port "${socket_port}" &
done