#! /usr/bin/env python

import json
import os
import re
from optparse import OptionParser

import cv2

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("", "--estimated-coordinate-file", dest="estimated_coordinate_file", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")

    (options, args) = parser.parse_args()

    received_json = {}
    frame_index = 0

    with open(options.estimated_coordinate_file) as fin_est_coordinates:
        received_json = json.load(fin_est_coordinates)

    print "Length of Annotations:", len(received_json)
    for j in range(0, len(received_json)):

        frame_index += 1
        annotation = received_json[str(j)]
        frame_file = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        frame = cv2.imread(frame_file)

        if (options.display_level >= 1):
            display_detection = frame.copy()

        for di in range(0, len(annotation["detections"])):
            tbp = annotation["detections"][di]["test_bodypart"]
            fi = annotation["detections"][di]["frame_index"]
            # display_detection /= 255.0
            cv2.circle(display_detection, (annotation["detections"][di]["coord_x"],annotation["detections"][di]["coord_y"]), 4, (0, di*255, 255), thickness=-1)

        if (options.display_level >= 1):
            cv2.imshow("Detected Frame", display_detection)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

