#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp '../expts/figures'

#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=200 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=100.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20160505_213143_fragmented/positive.p"  --negative-training-datafile="../expts/20160505_213143_fragmented/negative.p"
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=200 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20160505_213143_fragmented/positive.p"  --negative-training-datafile="../expts/20160505_213143_fragmented/negative.p"

#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170228_070510_fragmented_TrainAndTestOnSame File/positive.p"  --negative-training-datafile="../expts/20170228_070510_fragmented_TrainAndTestOnSame File/negative.p"

./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170228_070510_fragmented_TrainAndTestOnSame File/positive.p"  --negative-training-datafile="../expts/20170228_070510_fragmented_TrainAndTestOnSame File/negative.p"