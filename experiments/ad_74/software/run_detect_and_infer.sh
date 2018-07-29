#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp '../expts/figures'

./detect_and_infer.py --train-annotation-list-all="../config/annotation_list_old_new_all_forStratified" --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=200 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20160505_213143_fragmented/positive.p"  --negative-training-datafile="../expts/20160505_213143_fragmented/negative.p"