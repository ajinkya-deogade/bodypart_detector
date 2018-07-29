#!/bin/bash

DIR_ROOT_TRAIN="/Volumes/Macintosh HD/Ajinkya/mhdo/experiments/ad_56/expts/fpga/20160120_Hessian_250_nOctaves_2_nOctaveLayers_3/fragmented"

mkdir -vp ../expts/fpga/20160120_Hessian_250_nOctaves_2_nOctaveLayers_3/combined/
./combine_train_files_pos.py --train-feature-list ../config/combine_train_data_pos --save-file ../expts/fpga/20160120_Hessian_250_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_all.p --root-folder "${DIR_ROOT_TRAIN}/"

#./combine_train_files_neg.py --train-feature-list ../config/combine_train_data_neg --save-file ../expts/fpga/20160120_Hessian_250_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_all.p --root-folder "${DIR_ROOT_TRAIN}/"
