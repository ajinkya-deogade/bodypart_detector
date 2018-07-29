#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#for socket_port in `seq 9998 9999`
#do
#    ./bodypart_detector_server.py --positive-training-datafile ../expts/20150401_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_MouthHook_all.p --negative-training-datafile ../expts/20150401_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 1 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 0
#    ./bodypart_detector_server.py --positive-training-datafile ../expts/20150401_FPGA_TrainData/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_RightMHhook_all.p --negative-training-datafile ../expts/20150401_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_RightMHhook_all.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1
#    ./bodypart_detector_server.py --positive-training-datafile ../expts/20150401_FPGA_TrainData/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_LeftMHhook_all.p --negative-training-datafile ../expts/20150401_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_LeftMHhook_all.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1
#    ./bodypart_detector_server.py --positive-training-datafile ../expts/20150401_FPGA_TrainData/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_RightDO_all.p --negative-training-datafile ../expts/20150401_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_RightDO_all.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1
#    ./bodypart_detector_server.py --positive-training-datafile ../expts/20150401_FPGA_TrainData/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_LeftDO_all.p --negative-training-datafile ../expts/20150401_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_LeftDO_all.p --desc-dist-threshold -0.01 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1
#done

./bodypart_detector_server.py --positive-training-datafile ../expts/20150407_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_MouthHook_all.p --negative-training-datafile ../expts/20150407_FPGA_TrainData/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_FPGA_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --socket-port 9998 --opencv-keypoints 1