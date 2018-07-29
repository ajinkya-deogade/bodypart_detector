#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

for socket_port in {9988..9999}
do
    ./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_MouthHook_1.p --negative-training-datafile ../expts/train_neg_MouthHook_1.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0.0 --nthread 1 --verbosity 0 --socket-port "${socket_port}" &
done