#!/bin/bash

set -e
set -u
# set -x

source ./sourceme

time ./bodypart_detector_client.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --n-server 4 --test-bodypart LeftMHhook



