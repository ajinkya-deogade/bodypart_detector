#! /bin/bash

set -e
set -u
set -x

source ./sourceme

./train_detect_StratifiedShuffleSplit.py --train-annotation-list-all ../config/annotation_list_old_new_all_forStratified --project-path "${PROJECT_PATH}/" --mh-neighborhood 100 --display 0 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 250  --outlier-error-dist 15 --training-bodypart MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan