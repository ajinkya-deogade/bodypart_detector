#! /usr/bin/env python

from optparse import OptionParser
import json
import os
import struct
import cv2
import numpy as np
import socket
import sys
import copy
from scipy import interpolate

def main(options, args):

    bodypart_coords_new = {}

    test_bodypart = options.test_bodypart

    with open(options.start_end_file) as fin_StartEnd:
        bodypart_coords = dict(json.load(fin_StartEnd))

    frame_index = -1

    interpolatedCoordinateFile = os.path.join('../expts/interpolated_coordinates_' + test_bodypart.strip() + ".json")
    fileWriter_interpolatedCoordinates = open(interpolatedCoordinateFile, 'w+')

    for videonumber in range(0, len(bodypart_coords)):

        video_file = bodypart_coords[str(videonumber)]["VideoFile"]
        print "Video File: ", video_file
        cap = cv2.VideoCapture(video_file)

        bodypart_coords_new[videonumber] = {}
        bodypart_coords_new[videonumber]["VideoFile"] = video_file
        bodypart_coords_new[videonumber]["Coordinates"] = {}
        bodypart_coords_new[videonumber]["Frame_Sequence"] = {}

        n_start = len(bodypart_coords[str(videonumber)]["StartFrames"])
        print n_start
        n_end = len(bodypart_coords[str(videonumber)]["EndFrames"])
        # n_bodypart_visible_seq = 0

        if n_start == n_end:
            n_bodypart_visible_seq = n_start
            print "Number of Start and End points for video: ",n_bodypart_visible_seq
        else:
            print "Number of Start and End points not equal for video: ", str(videonumber),";  ", video_file
            continue

        if (cap.isOpened()):
            for n_seq in range(0, n_bodypart_visible_seq):
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq] = {}
                for i in range(0, len(bodypart_coords[str(videonumber)])-3):
                    if int(bodypart_coords[str(videonumber)][str(i)]["FrameIndexVideo"]) == int(bodypart_coords[str(videonumber)]["StartFrames"][n_seq]):
                        i_start = i
                    if int(bodypart_coords[str(videonumber)][str(i)]["FrameIndexVideo"]) == int(bodypart_coords[str(videonumber)]["EndFrames"][n_seq]):
                        i_end = i
                seqLength_index =  i_end - i_start

                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["StartFrame"] = bodypart_coords[str(videonumber)][str(i_start)]["FrameIndexVideo"]
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["EndFrame"] = bodypart_coords[str(videonumber)][str(i_end)]["FrameIndexVideo"]

                for n in range(i_start, i_end):
                    annotation_interval = bodypart_coords[str(videonumber)][str(n+1)]["FrameIndexVideo"] - bodypart_coords[str(videonumber)][str(n)]["FrameIndexVideo"]

                    x_endpoints = [bodypart_coords[str(videonumber)][str(n)]["x"], bodypart_coords[str(videonumber)][str(n+1)]["x"]]
                    y_endpoints = [bodypart_coords[str(videonumber)][str(n)]["y"], bodypart_coords[str(videonumber)][str(n+1)]["y"]]

                    interp_func = interpolate.interp1d(x_endpoints, y_endpoints)

                    if x_endpoints[0] == x_endpoints[1]:
                        x_interp = np.linspace(x_endpoints[0], x_endpoints[1], annotation_interval+1)
                        x_interp = [int(x) for x in x_interp]
                        y_interp = np.linspace(y_endpoints[0], y_endpoints[1], annotation_interval+1)
                        y_interp = [int(y) for y in y_interp]

                    else:
                        x_interp = np.linspace(x_endpoints[0], x_endpoints[1], annotation_interval+1)
                        x_interp = [int(x) for x in x_interp]
                        y_interp = interp_func(x_interp)
                        y_interp = [int(y) for y in y_interp]

                    local_iter = -1

                    for frameIndex in range(int(bodypart_coords[str(videonumber)][str(n)]["FrameIndexVideo"]), int(bodypart_coords[str(videonumber)][str(n+1)]["FrameIndexVideo"])):
                        cap.set(1, float(frameIndex))
                        ret,frame = cap.read()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex] = {}
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"] = {}

                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"] = {}
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["FrameIndexVideo"] = frameIndex
                        if (not ret):
                            continue

                        local_iter += 1
                        frame_index += 1

                        if frameIndex == bodypart_coords[str(videonumber)][str(n)]["FrameIndexVideo"]:
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["x"] = bodypart_coords[str(videonumber)][str(n)]["x"]
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["y"] = bodypart_coords[str(videonumber)][str(n)]["y"]
                        elif frameIndex == bodypart_coords[str(videonumber)][str(n+1)]["FrameIndexVideo"]:
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["x"] = bodypart_coords[str(videonumber)][str(n+1)]["x"]
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["y"] = bodypart_coords[str(videonumber)][str(n+1)]["y"]

                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"]["x"] = x_interp[local_iter]
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"]["y"] = y_interp[local_iter]

    json.dump(bodypart_coords_new, fileWriter_interpolatedCoordinates, sort_keys=True, indent=4, separators=(',', ': '))
    fileWriter_interpolatedCoordinates.close()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--start-end-file", dest="start_end_file", default="", help="path containing data directory")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--bodypart", dest="test_bodypart", default="MouthHook", help="Input the bodypart to be tested")

    (options, args) = parser.parse_args()

    main(options, args)


