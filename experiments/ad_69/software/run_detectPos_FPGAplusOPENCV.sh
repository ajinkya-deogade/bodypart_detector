#! /bin/bash

set -e
set -u
set -x

source ./sourceme

./detectPos_FPGAplusOPENCV.py --train-annotation-list-all ../config/annotation_list_old_new_all_forStratified --project-path "${PROJECT_PATH}/" --mh-neighborhood 100 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 150 --training-bodypart MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan --pos-neg-equal 1 --desc-dist-threshold 0 --vote-patch-size 10 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 10 --crop-size 256  --display 3