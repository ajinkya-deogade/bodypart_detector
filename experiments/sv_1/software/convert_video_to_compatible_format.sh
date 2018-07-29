#!/bin/bash

set -e
set -u
set -x

ffmpeg  -i $1 -c mpeg4 -f mp4 -q 0 $2
