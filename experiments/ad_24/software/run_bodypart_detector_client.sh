#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./bodypart_detector_continuousframes_client.py --start-end-file ../config/annotated_positions_RightDorsalOrgan.json --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 10 --test-bodypart RightDorsalOrgan