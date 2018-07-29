#!/bin/bash

set -e
set -u
set -x

./track_surf.py --file ~/work/CRG/data/12_20140213R_2.mp4 --bbox '(825,1300,125,125)'
