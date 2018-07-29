#!/bin/bash

set -e
set -u
set -x

mkdir -vp ../expts

./ex_MHhook_detection_train.py --train-annotation-list //Users/agomez/work/dev/mhdo/experiments/ad_3/config/train_annotation_list --project-path /Volumes/HD2/MHDO_Tracking/ --mh-neighborhood 50 --display 1 --training-datafile ../expts/data1.p --training-bodypart RightMHhook
./ex_MHhook_detection_test.py --test-annotation /Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames_MHhooks/10_20140214R_Frames_20140514_184730/10_20140214R_20140514_184730_Coordinates.json --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/data1.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images/10_20140214R_Frames_20140514_184730 --save-dir-error /Users/agomez/work/data/saved_error_new/10_20140214R_Error_20140514_184730.json
./ex_MHhook_detection_test.py --test-annotation /Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames_MHhooks/13_20140214R_Frames_20140514_225622/13_20140214R_20140514_225622_Coordinates.json --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/data1.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images/13_20140214R_Frames_20140514_225622 --save-dir-error /Users/agomez/work/data/saved_error_new/13_20140214R_Error_20140514_225622.json
./ex_MHhook_detection_test.py --test-annotation /Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames_MHhooks/14_20140214R_Frames_20140514_230436/14_20140214R_20140514_230436_Coordinates.json --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/data1.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images/14_20140214R_Frames_20140514_230436 --save-dir-error /Users/agomez/work/data/saved_error_new/14_20140214R_Error_20140514_230436.json
./ex_MHhook_detection_test.py --test-annotation /Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames_MHhooks/15_20140213R_Frames_20140514_231415/15_20140213R_20140514_231415_Coordinates.json --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/data1.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images/15_20140213R_Frames_20140514_231415 --save-dir-error /Users/agomez/work/data/saved_error_new/15_20140213R_Error_20140514_231415.json
./ex_MHhook_detection_test.py --test-annotation /Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames_MHhooks/15_20140214R_Frames_20140514_232059/15_20140214R_20140514_232059_Coordinates.json --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/data1.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images/15_20140214R_Frames_20140514_232059 --save-dir-error /Users/agomez/work/data/saved_error_new/15_20140214R_Error_20140514_232059.json
./ex_MHhook_detection_test.py --test-annotation /Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames_MHhooks/16_20140213R_Frames_20140514_233042/16_20140213R_20140514_233042_Coordinates.json --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/data1.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images/16_20140213R_Frames_20140514_233042 --save-dir-error /Users/agomez/work/data/saved_error_new/16_20140213R_Error_20140514_233042.json
./ex_MHhook_detection_test.py --test-annotation /Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/Extracted_Frames_MHhooks/16_20140214R_Frames_20140514_233743/16_20140214R_20140514_233743_Coordinates.json --project-path /Volumes/HD2/MHDO_Tracking/ --training-data-file ../expts/data1.p --desc-dist-threshold 0.4 --vote-patch-size 10 --vote-sigma 5 --outlier-error-dist 15 --display 0 --save-dir-images /Users/agomez/work/data/saved_images/16_20140214R_Frames_20140514_233743 --save-dir-error /Users/agomez/work/data/saved_error_new/16_20140214R_Error_20140514_233743.json

cd ..
cd ..
cd sv_5/software/
./run_ex_mouth_hook_detection_NNvote.sh


