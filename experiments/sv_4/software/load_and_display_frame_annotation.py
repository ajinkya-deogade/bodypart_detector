#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-a", "--annotation", dest="annotation_file", default="", help="frame level annotation JSON file")
    parser.add_option("-p", "--project-path", dest="project_dir", default="", help="path containing data directory")

    (options, args) = parser.parse_args()

    print "annotation_file:" , options.annotation_file

    with open(options.annotation_file) as fin_annotation:
        annotation = json.load(fin_annotation)

    for i in range(0, len(annotation["Annotations"])):
        frame_file = annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir , frame_file)
        print frame_file

        frame = cv2.imread(frame_file)

        display_frame = frame.copy()
        
        mh_coords = None
        for j in range(0, len(annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook"):
                mh_coords = (int(annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]), int(annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))

        if (mh_coords is not None):
            cv2.circle(display_frame, mh_coords, 10, (0, 0, 255), thickness=-1)

        display_frame = cv2.resize(display_frame, (0,0), fx=0.25, fy=0.25)
        cv2.imshow("frame", display_frame)
        cv2.waitKey(1000)

    cv2.destroyWindow("frame")
