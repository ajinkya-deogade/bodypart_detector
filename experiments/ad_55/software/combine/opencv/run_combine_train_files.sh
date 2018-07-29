#!/bin/bash

mkdir -vp ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined_2/
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_MH_pos_2 --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined_2/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_MH_neg_2 --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined_2/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p
echo "Mouth Hook Done....."
sleep 60s

./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightMHhook_pos --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightMHhook_neg --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p
echo "Right Mouth Hook Done....."
sleep 20s
#
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftMHhook_pos --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftMHhook_neg --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p
echo "Left Mouth Hook Done....."
sleep 20s
#
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftDO_pos --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftDO_neg --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p
echo "Left DO Done....."
sleep 20s
#
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightDO_pos --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p
./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightDO_neg --save-file ../../../expts/opencv/20150519_Hessian_500_nOctaves_2_nOctaveLayers_3/combined/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p
echo "Right DO Done....."
sleep 20s

#mkdir -vp ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_MH_pos --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_600_nOctaves_2_nOctaveLayers_3_MouthHook_all.p
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_MH_neg --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_600_nOctaves_2_nOctaveLayers_3_MouthHook_all.p
#
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightMHhook_pos --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_600_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightMHhook_neg --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_600_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p
#
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightDO_pos --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_600_nOctaves_2_nOctaveLayers_3_RightDO_all.p
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_RightDO_neg --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_600_nOctaves_2_nOctaveLayers_3_RightDO_all.p
#
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftDO_pos --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_600_nOctaves_2_nOctaveLayers_3_LeftDO_all.p
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftDO_neg --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_600_nOctaves_2_nOctaveLayers_3_LeftDO_all.p
#
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftMHhook_pos --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_pos_Hessian_600_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p
#./combine_train_files.py --train-feature-list ../../../config/forCombiningTrainingData/opencv/opencv/combine_train_data_list_LeftMHhook_neg --save-file ../../../expts/20150519_Hessian_600_nOctaves_2_nOctaveLayers_3/train_neg_Hessian_600_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p