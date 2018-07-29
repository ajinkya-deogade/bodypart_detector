#!/bin/bash

set -e
set -u
set -x


source ./sourceme

mkdir -vp ../expts

./distance_moving_MH.py --bodypart-1 RightDorsalOrgan --bodypart-2 LeftDorsalOrgan --interpolated_file_bodypart_1 ../expts/interpolated_coordinates_RightDorsalOrgan.json --interpolated_file_bodypart_2 ../expts/interpolated_coordinates_LeftDorsalOrgan.json --interpolated_file_MH ../expts/interpolated_coordinates_MouthHook.json --project-path "${PROJECT_PATH}/"