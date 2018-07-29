#!/bin/bash

set -e
set -u
set -x


source ./sourceme

mkdir -vp ../expts

./simple_interpolation.py --start-end-file ../expts/end_start_positions_MouthHook.json --project-path "${PROJECT_PATH}/" --bodypart MouthHook
