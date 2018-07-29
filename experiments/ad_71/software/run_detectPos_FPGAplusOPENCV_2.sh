#! /bin/bash

set -e
set -u
set -x

source ./sourceme

./detectPos_FPGAplusOPENCV_2.py --train-annotation-list-all ../config/annotation_list_old_new_all_forStratified --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 200 --training-bodypart MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 10 --crop-size 256  --display 0