#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp '../expts/figures'

./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=250 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/train_pos_Hessian_250_nOctaves_2_nOctaveLayers_3_all.p"
b
