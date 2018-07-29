#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp '../expts/figures'

#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=200 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=100.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20160505_213143_fragmented/positive.p"  --negative-training-datafile="../expts/20160505_213143_fragmented/negative.p"
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=200 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20160505_213143_fragmented/positive.p"  --negative-training-datafile="../expts/20160505_213143_fragmented/negative.p"

#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20160505_213143_fragmented/positive.p"  --negative-training-datafile="../expts/20160505_213143_fragmented/negative.p"

## For Training The Complete Set - Pos Neg Equal
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170224_150128_CompleteSet_PosNegEqual/positive.p"  --negative-training-datafile="../expts/20170224_150128_CompleteSet_PosNegEqual/negative.p" --output-folder "20170224_150128_CompleteSet_PosNegEqual"

## For Training The Complete Set  - Pos Neg Not Equal
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170224_153728_CompleteSet_PosNegNotEqual/positive.p"  --negative-training-datafile="../expts/20170224_153728_CompleteSet_PosNegNotEqual/negative.p" --output-folder "20170224_153728_CompleteSet_PosNegNotEqual"

## For Training The Complete Set - Clips  - Pos Neg Equal
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170224_160946_WithoutClips_PosNegEqual/positive.p"  --negative-training-datafile="../expts/20170224_160946_WithoutClips_PosNegEqual/negative.p" --output-folder "20170224_160946_WithoutClips_PosNegEqual"

## For Training The Complete Set - Clips  - Pos Neg Not Equal
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170224_163940_WithoutClips_PosNegNotEqual/positive.p"  --negative-training-datafile="../expts/20170224_163940_WithoutClips_PosNegNotEqual/negative.p" --output-folder "20170224_163940_WithoutClips_PosNegNotEqual"

## For Training on Dummy Train Data
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,RightDorsalOrgan,LeftDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170224_175736_fragmented/positive.p"  --negative-training-datafile="../expts/20170224_175736_fragmented/negative.p" --output-folder "20170224_150128_CompleteSet_PosNegEqual"

## Train and Test on Same
#./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170228_180100_fragmented/positive.p"  --negative-training-datafile="../expts/20170228_180100_fragmented/negative.p" --output-folder "20170224_150128_sameTrainAndTest_PosNegEqual"

## Train using new_with_clips and Test on 20170303_205031
./detect_and_infer.py --mh-neighborhood=50 --nOctaves=2 --nOctaveLayers=3 --hessian-threshold=150 --training-bodypart="MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan" --desc-dist-threshold=0 --vote-patch-size=7 --vote-sigma=5 --vote-threshold=6.0 --outlier-error-dist=10 --crop-size=256 --display=0  --project-path="${PROJECT_PATH}/" --positive-training-datafile="../expts/20170303_050546_onlyNewWithClips_RemovedDuplicates/python/positive.p"  --negative-training-datafile="../expts/20170303_050546_onlyNewWithClips_RemovedDuplicates/python/negative.p" --output-folder "Rawdata_20170303_205031"