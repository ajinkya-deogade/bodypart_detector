#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

#time ./bodypart_detector_client.py --test-annotation-list ../config/train_annotation_list_old_new_all_train --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images "E:/MHDO_Tracking/data/20150716/train/"
#time ./bodypart_detector_client.py --test-annotation-list ../config/train_annotation_list_old_new_all_test --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images "E:/MHDO_Tracking/data//20150716/test/"
#time ./bodypart_detector_client.py --test-annotation-list ../config/test_descriptors --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images "F:/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/FPGA/test_descriptors/"

#time ./bodypart_detector_client.py --test-annotation-list '../config/annotation_list_old_new_all_forStratified' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/20170225_CroppedImage_CompleteSet/'

#time ./bodypart_detector_client.py --test-annotation-list '../config/temp' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/temp/'

#time ./bodypart_detector_client.py --test-annotation-list '../config/temp' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/temp/'


#time ./bodypart_detector_client.py --test-annotation-list '../config/temp_2' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/temp/'
#time ./bodypart_detector_client.py --test-annotation-list '../config/annotation_list_only_old_clips' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/annotation_list_only_old_clips/'

#time ./bodypart_detector_client.py --test-annotation-list '../config/trainUsingSameDay_20170224Data' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/trainUsingSameDay_20170224Data/'

#time ./bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20170303' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170303/'
#time ./bodypart_detector_client.py --test-annotation-list '../config/dataAnnotatedOn_20170306' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataAnnotatedOn_20170306/'
#time ./bodypart_detector_client.py --test-annotation-list '../config/trainAndTestOnSame_201605_start_2600' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/trainAndTestOnSame_201605_start_2600/'


#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20170316' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170316/'
#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20170317' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170317/'
#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20170318' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170318/'
#time python bodypart_detector_client.py --test-annotation-list '../config/Rawdata_20170317_233847' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170318/'
#time python bodypart_detector_client.py --test-annotation-list '../config/Rawdata_20170318_190126' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170318/'

#time python bodypart_detector_client.py --test-annotation-list '../config/temp' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170318/'

#time python bodypart_detector_client.py --test-annotation-list '../config/temp' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170318/'

#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20180417' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20180417/'

#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20180417' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20180417/'

#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20180417_re' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20180417_re/'

#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20180417_test' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20180417_test/'

#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20170317_re' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170317_re/'

#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20180417_re' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20180417_re/'

#time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20170318_re' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20170318_re'

time python bodypart_detector_client.py --test-annotation-list '../config/dataCollectedOn_20180417_re2' --project-path "${PROJECT_PATH}/" --display 0 --detect-bodypart MouthHook --verbosity 1 --save-dir-images '../expts/dataCollectedOn_20180417_re2/'