#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../../../expts

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/opencv/20150517_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000



########################### For FPGA Training Data ##########################
# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150608_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

                                            ##### 20150609 #####

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150609_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

##### 20150617 #####

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150617_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150618_2 #############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150618_2_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150702 #############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150702 OpenCV Orientation#############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150702 FPGA Orientation#############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150702_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150706 OpenCV Orientation Hessian 500 #############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150706 FPGA Orientation Hessian 500 #############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150706 OpenCV Orientation Hessian 250 #############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150706 FPGA Orientation Hessian 250 #############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150706_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150707_2 FPGA Orientation Hessian 500 #############################################

# Mouth Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000

################################# 20150708 FPGA Orientation Hessian 250 #############################################

# Mouth Hook
./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --negative-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 10000

# Left MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 11000

# Right MH-Hook
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --negative-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0  --verbosity 1 --socket-port 12000

# Left Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --negative-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_LeftDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 13000

# Right Dorsal Organ
#./bodypart_detector_server_fpga.py --positive-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_RightDO_all.p --negative-training-datafile ../../../expts/fpga/20150708_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_RightDO_all.p --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --display 0 --vote-threshold 0 --nthread 1 --opencv 0 --verbosity 1 --socket-port 14000