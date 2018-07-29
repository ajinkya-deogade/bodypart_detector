#!/bin/bash

set -e
set -u
set -x

source ./sourceme

########################## Hessian 500 #################################
#
#DIR_KEYPOINTS=F:/FPGA_Validation/20150706_500_2/Train/validKeyPoints/
#DIR_DESCRIPTORS=F:/FPGA_Validation/20150706_500_2/Train/descriptors/

#mkdir -vp ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_3 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_4 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_5 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_6 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"

#mkdir -vp ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_3 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_4 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_5 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
#./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_6 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 500 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_500_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_500_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"

########################## Hessian 250 #################################

DIR_KEYPOINTS=/Volumes/HD2/FPGA_Validation/20150708_250/Train/validKeyPoints/
DIR_DESCRIPTORS=/Volumes/HD2/FPGA_Validation/20150708_250/Train/descriptors/
##
##mkdir -vp ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/
##./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
##./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
##./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_3 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
##./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_4 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
##./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_5 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
##./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_6 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --negative-training-datafile ../../../expts/fpga/20150707_2_Hessian_250_nOctaves_2_nOctaveLayers_3_opencvOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"

mkdir -vp ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/
./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_1 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --negative-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_1.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_2 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --negative-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_2.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_3 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --negative-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_3.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_4 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --negative-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_4.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_5 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --negative-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_5.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"
./ex_MHhook_detection_train_fpga.py --train-annotation-list ../../../config/forTraining/train_annotation_list_MH_6 --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250 --positive-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --negative-training-datafile ../../../expts/fpga/20150826_Hessian_250_nOctaves_2_nOctaveLayers_3_fpgaOrientation/fragmented/train_neg_Hessian_250_nOctaves_2_nOctaveLayers_3_MouthHook_6.p --training-bodypart MouthHook --dir-keypoints "${DIR_KEYPOINTS}/" --dir-descriptor "${DIR_DESCRIPTORS}/"