#! /bin/bash

set -e
set -u
set -x

source ./sourceme

#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170222_forValidation/Rawdata_20170222_171305/Rawdata_20170222_171305.mp4' --meta-data-folder '20170224_150128_CompleteSet_PosNegEqual'
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170222_forValidation/Rawdata_20170222_171305/Rawdata_20170222_171305.mp4' --meta-data-folder '20170224_153728_CompleteSet_PosNegNotEqual'

#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_154914/Rawdata_20170224_154914.mp4' --meta-data-folder '20170224_150128_CompleteSet_PosNegEqual'
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_154914/Rawdata_20170224_154914.mp4' --meta-data-folder '20170224_153728_CompleteSet_PosNegNotEqual'
#
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_155228/Rawdata_20170224_155228.mp4' --meta-data-folder '20170224_150128_CompleteSet_PosNegEqual'
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_155228/Rawdata_20170224_155228.mp4' --meta-data-folder '20170224_153728_CompleteSet_PosNegNotEqual'
#
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_155514/Rawdata_20170224_155514.mp4' --meta-data-folder '20170224_150128_CompleteSet_PosNegEqual'
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_155514/Rawdata_20170224_155514.mp4' --meta-data-folder '20170224_153728_CompleteSet_PosNegNotEqual'
#
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_160450/Rawdata_20170224_160450.mp4' --meta-data-folder '20170224_150128_CompleteSet_PosNegEqual'
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_160450/Rawdata_20170224_160450.mp4' --meta-data-folder '20170224_153728_CompleteSet_PosNegNotEqual'
#
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_160846/Rawdata_20170222_171305.mp4' --meta-data-folder '20170224_150128_CompleteSet_PosNegEqual'
#./stampDetections.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_160846/Rawdata_20170222_171305.mp4' --meta-data-folder '20170224_153728_CompleteSet_PosNegNotEqual'

#./stampDetections_oneFrame.py --video-file '/users/Ajinkya/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170224_forValidation/Rawdata_20170224_154914/Rawdata_20170224_154914.mp4' --meta-data-folder '20170228_TrainAndTestOnSameFile'

#./stampDetections_oneFrame.py --video-file '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/Gaussian/Rawdata_20170303_205031/Rawdata_20170303_205031.mp4' --meta-data-folder 'Rawdata_20170303_205031'

#./stampDetections_oneFrame_withSpot.py --video-file-list ../config/temp --project-path "${PROJECT_PATH}/" --meta-data-folder 'Rawdata_20170303_205031' --plot-body-part 'MouthHook'

#./stampDetections_oneFrame_withSpot_MultipleBP.py --video-file-list ../config/trainAndTestOnSame_205031 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan'
#./stampDetections_oneFrame_withSpot_MultipleBP.py --video-file-list ../config/trainAndTestOnSame_201605_start_2600 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan'
#./stampDetections_oneFrame_withSpot_MultipleBP.py --video-file-list ../config/trainAndTestOnSame_205257_start_4000 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan'
#./stampDetections_oneFrame_withSpot_MultipleBP.py --video-file-list ../config/trainAndTestOnSame_210131_start_1000 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan'

#python stampDetections_oneFrame_withSpot_MultipleBP.py --video-file-list ../config/Rawdata_20170315_181142 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan'

#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_210553 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_182901 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_181142 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_202334 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_181858 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_181142 --project-path "${PROJECT_PATH}/" --plot-body-part 'Head' --spot-size 200
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_210553 --project-path "${PROJECT_PATH}/" --plot-body-part 'Head' --spot-size 200
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170315_182901 --project-path "${PROJECT_PATH}/" --plot-body-part 'Head' --spot-size 200


#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170311_215549 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170311_221257 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170311_222656 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170311_223804 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170311_214541 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50

python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170316_181916 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 75
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170316_182806 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170316_190059 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50
#python stampDetections_MultipleBP_confThresh.py --video-file-list ../config/Rawdata_20170316_210613 --project-path "${PROJECT_PATH}/" --plot-body-part 'MouthHook,LeftDorsalOrgan,RightDorsalOrgan' --spot-size 50