#! /usr/bin/env python

from optparse import OptionParser
import cv2
import re
import copy
import numpy as np

r_coords = re.compile(r'\((?P<x>[^,]+),(?P<y>[^,]+),(?P<w>[^,]+),(?P<h>.+)\)')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--file", dest="video_file", default="", help="video file to play")
    parser.add_option("", "--bbox", dest="bounding_box", default="", help="bounding box to be tracked (x,y,w,h)")

    (options, args) = parser.parse_args()

    print "video_file:" , options.video_file

    coords_match = r_coords.match(options.bounding_box)

    bbox_init = {}
    if (coords_match != None):
        bbox_init["x1"] = int(coords_match.group("x"))
        bbox_init["y1"] = int(coords_match.group("y"))
        bbox_init["w"] = int(coords_match.group("w"))
        bbox_init["h"] = int(coords_match.group("h"))
        bbox_init["x2"] = int(coords_match.group("w")) + bbox_init["x1"]
        bbox_init["y2"] = int(coords_match.group("h")) + bbox_init["y1"]
    else:
        print "Error parsing bounding box from input argument"
        exit

    video_in = cv2.VideoCapture(options.video_file)
    fps = video_in.get(cv2.cv.CV_CAP_PROP_FPS)
    print "fps:" , fps

    frame_dur = int(1000.0 / float(fps))
    print "frame duration (ms):" , frame_dur

    bbox_prev = copy.deepcopy(bbox_init)
    bbox = copy.deepcopy(bbox_init)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    frame_id = 0
    while (True):
        ret, frame = video_in.read()

        if (ret == False):
            break

        frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
        frame = clahe.apply(frame)

        if (frame_id >= 1):
            template = frame_prev[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
            cv2.imshow("Template", template)

            res = cv2.matchTemplate(frame[bbox["y1"]-100:bbox["y2"]+100, bbox["x1"]-100:bbox["x2"]+100], template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            bbox["x1"] += max_loc[0] - 100
            bbox["y1"] += max_loc[1] - 100
            bbox["x2"] = bbox["x1"] + bbox["w"]
            bbox["y2"] = bbox["y1"] + bbox["h"]

        frame_prev = frame.copy()

        display_image = frame.copy()
        print "bbox: " , bbox
        cv2.rectangle(display_image, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 255, 0), 2)

        display_image = cv2.resize(display_image, (0,0), fx=0.25, fy=0.25)
        cv2.imshow(options.video_file, display_image)

        if (frame_id == 0 and False):
            k = cv2.waitKey(-1)
        else:
            k = cv2.waitKey(3)

        if (k == 'q'):
            exit

        frame_id = frame_id + 1

    video_in.release()
