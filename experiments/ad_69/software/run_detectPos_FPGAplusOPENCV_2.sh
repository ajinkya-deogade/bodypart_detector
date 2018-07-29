#! /bin/bash

set -e
set -u
set -x

source ./sourceme

./detectPos_FPGAplusOPENCV_2.py --test-annotation-list '../config/test_annotation_list_fpgaKNNVal' --project-path "${PROJECT_PATH}/" --mh-neighborhood 50 --nOctaves 2 --nOctaveLayers 3 --hessian-threshold 150 --training-bodypart MouthHook --pos-neg-equal 1 --desc-dist-threshold 0 --vote-patch-size 7 --vote-sigma 5 --vote-threshold 0 --outlier-error-dist 10 --crop-size 256  --display 3