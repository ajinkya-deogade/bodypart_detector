#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_RightDO.p --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 10000
#./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_RightMHhook.p --negative-training-datafile ../expts/train_neg_RightMHhook.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 10000
#./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_MouthHook.p --negative-training-datafile ../expts/train_neg_MouthHook.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 10000