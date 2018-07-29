#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

time ./bodypart_detector_client.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 10  --detect-bodypart LeftDorsalOrgan,RightDorsalOrgan --verbosity 1



