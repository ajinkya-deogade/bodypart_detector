#!/bin/bash

set -e
set -u
set -x

source ./sourceme

mkdir -vp ../expts
mkdir -vp ../expts/20140826

<<<<<<< HEAD
./bodypart_detector_client_2.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --test-bodypart LeftDorsalOrgan
=======
./bodypart_detector_client_2.py --test-annotation-list ../config/test_annotation_list --project-path "${PROJECT_PATH}/" --display 0 --outlier-error-dist 15 --test-bodypart RightDorsalOrgan --save-folder ../expts/20140826/
>>>>>>> e756cdad8c51ebb5a110591bfdbeba7e36532c80



