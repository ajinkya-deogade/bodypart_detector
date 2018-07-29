#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

<<<<<<< HEAD
./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_LeftDO.p  --negative-training-datafile ../expts/train_neg_LeftDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 9989
=======
./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_RightDO_1.p  --negative-training-datafile ../expts/train_neg_RightDO_1.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 9988
>>>>>>> e756cdad8c51ebb5a110591bfdbeba7e36532c80
#./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_RightMHhook.p  --negative-training-datafile ../expts/train_neg_RightMHhook.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 9988
#./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_MouthHook.p  --negative-training-datafile ../expts/train_neg_MouthHook.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 9988
#./bodypart_detector_server.py --positive-training-datafile ../expts/train_pos_RightDO.p  --negative-training-datafile ../expts/train_neg_RightDO.p --desc-dist-threshold 0.005 --vote-patch-size 10 --vote-sigma 5 --display 0 --socket-port 9988
