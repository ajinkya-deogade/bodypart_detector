#!/bin/bash

set -e
set -u
set -x

source ./sourceme

for socket_port in {9998..10001}
do
    ./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_LeftMHhook_1.p --negative-training-datafile ../expts/train_neg_LeftMHhook_1.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --verbosity 1 --socket-port "${socket_port}" &
done