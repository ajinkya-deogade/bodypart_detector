#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts

# Mouth Hook
#for socket_port in {9998..10001}
#do
#    ./bodypart_detector_server_opencv.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 1  --verbosity 1 --socket-port 9998
#    ./bodypart_detector_server_opencv.py --positive-training-datafile ../../../expts/opencv/20150409_Hessian_500_nOctaves_3_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/opencv/20150409_Hessian_500_nOctaves_3_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_3_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 1  --verbosity 1 --socket-port 9998
#done

# Left MH-Hook
#for socket_port in 9998
#do
#./bodypart_detector_server_opencv.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 1  --verbosity 1 --socket-port 9998
#done

# Right MH-Hook
#for socket_port in 9998
#do
#./bodypart_detector_server_opencv.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 1  --verbosity 1 --socket-port 9998
#done

# Left Dorsal Organ
#for socket_port in 9998
#do
#./bodypart_detector_server_opencv.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv-keypoints 1 --verbosity 1 --socket-port 9998
#done

# Right Dorsal Organ
#for socket_port in 9998
#do
mkdir -vp ../expts/detected_images_rightDO/
./bodypart_detector_server_opencv.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 1 --vote-threshold 0 --nthread 1 --opencv-keypoints 1 --verbosity 1 --socket-port 9998 --save-dir ../expts/detected_images_rightDO/
#done