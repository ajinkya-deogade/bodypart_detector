#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../expts

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook.p --negative-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook.p --negative-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook.p --negative-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO.p --negative-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO.p --negative-training-datafile ../../expts/new/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000
