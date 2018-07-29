#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

for socket_port in `seq 9998 10007`
do
#    ./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_20141127.p --negative-training-datafile ../expts/train_neg_20141127.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --socket-port "${socket_port}" --detect-bodypart MouthHook --verbosity 1 &
    ./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_20141127.p --negative-training-datafile ../expts/train_neg_20141127.p --desc-dist-threshold 0.15 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 6.0 --nthread 1 --socket-port "${socket_port}" --detect-bodypart LeftDorsalOrgan,RightDorsalOrgan --verbosity 1 &
done


