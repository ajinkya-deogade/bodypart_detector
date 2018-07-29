#!/bin/bash

set -e
set -u
set -x

./load_and_display_frame_annotation.py --annotation ~/work/CRG/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames/13_20140214R_Frames_20140429_184448/13_20140214R_20140429_184448_Coordinates.json --project-path ~/work/CRG
