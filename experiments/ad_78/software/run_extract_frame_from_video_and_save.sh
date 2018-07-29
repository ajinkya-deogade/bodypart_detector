#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./extract_frame_from_video_and_save.py --test-annotation-list "../config/test_annotation_list_clips" --project-path "${PROJECT_PATH}/" --display 0



