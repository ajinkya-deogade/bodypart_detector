#!/bin/bash

set -e
set -u
set -x

set +u
export PYTHONPATH=/Users/Ajinkya/Downloads/opencv-master/samples/python:"${PYTHONPATH}"
set -u

./track_klt.py '/Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/Gaussian/Rawdata_20170303_205031/Rawdata_20170303_205031.mp4'
