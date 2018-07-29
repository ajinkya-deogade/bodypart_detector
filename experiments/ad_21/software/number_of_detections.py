#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import os
import re
import struct
import cv2
import numpy as np
import sys
import copy

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("", "--estimated-coordinate-file", dest="estimated_coordinate_file", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")

    (options, args) = parser.parse_args()

    bodypart_coords_est = {}
    frame_number = 0
    detected = 0
    est_coordinates = []
    with open(options.estimated_coordinate_file) as fin_est_coordinates:
        for line in fin_est_coordinates:
            try:
                est_coordinate = json.loads(line.rstrip())
                est_coordinates.append(est_coordinate)
            except ValueError:
                print "Not Read the Line", line
                pass

    print "Working on File: ", options.estimated_coordinate_file

    for i in range(0, len(est_coordinates)-1):
        frame_number += 1
        if "coord_x" in est_coordinates[i]["detections"][0]:
            detected += 1
    percentage_detected = float(float(detected)*float(100)/float(frame_number))
    print "Percentage Detected: %d/%d = %f"%(detected,frame_number,percentage_detected)