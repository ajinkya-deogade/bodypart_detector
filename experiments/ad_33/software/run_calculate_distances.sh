#!/bin/bash

set -e
set -u
set -x


source ./sourceme

mkdir -vp ../expts


./calculate_distances.py --train-annotation-list ../config/complete_list --project-path "${PROJECT_PATH}/" --training-bodypart-1 LeftDorsalOrgan --training-bodypart-2 RightDorsalOrgan
