#!/bin/bash

./combine_train_files.py --train-feature-list ../config/combine_train_data_list_MH_pos --save-file ../expts/20150326/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p
./combine_train_files.py --train-feature-list ../config/combine_train_data_list_MH_neg --save-file ../expts/20150326/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_all.p

./combine_train_files.py --train-feature-list ../config/combine_train_data_list_RightMHhook_pos --save-file ../expts/20150326/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p
./combine_train_files.py --train-feature-list ../config/combine_train_data_list_RightMHhook_neg --save-file ../expts/20150326/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightMHhook_all.p
./combine_train_files.py --train-feature-list ../config/combine_train_data_list_LeftMHhook_pos --save-file ../expts/20150326/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p
./combine_train_files.py --train-feature-list ../config/combine_train_data_list_LeftMHhook_neg --save-file ../expts/20150326/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftMHhook_all.p

./combine_train_files.py --train-feature-list ../config/combine_train_data_list_RightDO_pos --save-file ../expts/20150326/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p
./combine_train_files.py --train-feature-list ../config/combine_train_data_list_RightDO_neg --save-file ../expts/20150326/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_RightDO_all.p
./combine_train_files.py --train-feature-list ../config/combine_train_data_list_LeftDO_pos --save-file ../expts/20150326/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p
./combine_train_files.py --train-feature-list ../config/combine_train_data_list_LeftDO_neg --save-file ../expts/20150326/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_LeftDO_all.p