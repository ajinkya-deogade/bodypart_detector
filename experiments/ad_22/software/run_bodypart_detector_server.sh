#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts
#"${socket_port}" &
#for socket_port in  {9998..9998}
#do
./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --socket-port 9998
#done