#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0  --verbosity 1 --socket-port 9998

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0  --verbosity 1 --socket-port 9999

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0  --verbosity 1 --socket-port 10000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0 --verbosity 1 --socket-port 10001

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0 --verbosity 1 --socket-port 10002

########################### For FPGA Training Data ##########################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0  --verbosity 1 --socket-port 9998

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0  --verbosity 1 --socket-port 9999

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0  --verbosity 1 --socket-port 10000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0 --verbosity 1 --socket-port 10001

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 0 --verbosity 1 --socket-port 10002