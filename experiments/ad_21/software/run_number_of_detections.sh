#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts

./number_of_detections.py --estimated-coordinate-file E:/ExpResults/20140820_StabilityandFPS_Analysis_MouthHook/FPS=60/Results_20140820_181936.txt
./number_of_detections.py --estimated-coordinate-file E:/ExpResults/20140820_StabilityandFPS_Analysis_MouthHook/FPS=80/Results_20140820_182721.txt
./number_of_detections.py --estimated-coordinate-file E:/ExpResults/20140820_StabilityandFPS_Analysis_MouthHook/FPS=100/Results_20140820_183230.txt
./number_of_detections.py --estimated-coordinate-file E:/ExpResults/20140820_StabilityandFPS_Analysis_MouthHook/FPS=120/Results_20140820_183716.txt
./number_of_detections.py --estimated-coordinate-file E:/ExpResults/20140820_StabilityandFPS_Analysis_MouthHook/FPS=150/Results_20140820_184920.txt
./number_of_detections.py --estimated-coordinate-file E:/ExpResults/20140820_StabilityandFPS_Analysis_MouthHook/FPS=NoControl/Results_20140820_185241.txt