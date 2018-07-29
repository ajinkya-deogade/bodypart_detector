#!/bin/bash

set -e
set -u
set -x


source ./sourceme

mkdir -vp ../expts

./get_start_end_frames.py --annotation-list ../config/test_annotation_list_complete_clipNew --project-path "${PROJECT_PATH}/" --bodypart MouthHook