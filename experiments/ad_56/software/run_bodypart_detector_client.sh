#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

time ./bodypart_detector_client.py --test-annotation-list ../config/test_annotation_list_old_new --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 1 --detect-bodypart MouthHook,LeftMHhook,RightMHhook,LeftDorsalOrgan,RightDorsalOrgan --verbosity 0