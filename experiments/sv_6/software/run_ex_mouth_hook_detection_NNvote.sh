#!/bin/bash

set -e
set -u
set -x

mkdir -vp ../expts/5_20140213R_Frames_20140429_180427
./mouth_hook_detection_NNvote.py --train-annotation-list ../config/train_annotation_list --test-annotation ~/work/CRG/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames/5_20140213R_Frames_20140429_180427/5_20140213R_20140429_180427_Coordinates.json --project-path ~/work/CRG --mh-neighborhood 50 --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --nthread 8 --save-dir ../expts/5_20140213R_Frames_20140429_180427

mkdir -vp ../expts/5_20140214R_Frames_20140429_182312
./mouth_hook_detection_NNvote.py --train-annotation-list ../config/train_annotation_list --test-annotation ~/work/CRG/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames/5_20140214R_Frames_20140429_182312/5_20140214R_20140429_182312_Coordinates.json --project-path ~/work/CRG --mh-neighborhood 50 --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --nthread 8 --save-dir ../expts/5_20140214R_Frames_20140429_182312

mkdir -vp ../expts/7_20140214R_Frames_20140429_182822
./mouth_hook_detection_NNvote.py --train-annotation-list ../config/train_annotation_list --test-annotation ~/work/CRG/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames/7_20140214R_Frames_20140429_182822/7_20140214R_20140429_182822_Coordinates.json --project-path ~/work/CRG --mh-neighborhood 50 --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --nthread 8 --save-dir ../expts/7_20140214R_Frames_20140429_182822

mkdir -vp ../expts/9_20140214R_Frames_20140429_183426
./mouth_hook_detection_NNvote.py --train-annotation-list ../config/train_annotation_list --test-annotation ~/work/CRG/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames/9_20140214R_Frames_20140429_183426/9_20140214R_20140429_183426_Coordinates.json --project-path ~/work/CRG --mh-neighborhood 50 --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --nthread 8 --save-dir ../expts/9_20140214R_Frames_20140429_183426

mkdir -vp ../expts/10_20140214R_Frames_20140429_183916
./mouth_hook_detection_NNvote.py --train-annotation-list ../config/train_annotation_list --test-annotation ~/work/CRG/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames/10_20140214R_Frames_20140429_183916/10_20140214R_20140429_183916_Coordinates.json --project-path ~/work/CRG --mh-neighborhood 50 --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --nthread 8 --save-dir ../expts/10_20140214R_Frames_20140429_183916

mkdir -vp ../expts/13_20140214R_Frames_20140429_184448
./mouth_hook_detection_NNvote.py --train-annotation-list ../config/train_annotation_list --test-annotation ~/work/CRG/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames/13_20140214R_Frames_20140429_184448/13_20140214R_20140429_184448_Coordinates.json --project-path ~/work/CRG --mh-neighborhood 50 --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --nthread 8 --save-dir ../expts/13_20140214R_Frames_20140429_184448

