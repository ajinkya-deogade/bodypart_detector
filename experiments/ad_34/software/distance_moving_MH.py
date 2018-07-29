#! /usr/bin/env python

from optparse import OptionParser
import json
import os
import struct
import cv2
import numpy as np
import pylab as P
from operator import itemgetter
import socket
import sys
import copy
from scipy import interpolate

def main(options, args):

    bodypart_coords_new = {}

    bodypart_1 = options.bodypart_1
    bodypart_2 = options.bodypart_2

    with open(options.interpolated_file_MH) as fin_StartEnd:
        interpolated_MH = dict(json.load(fin_StartEnd))

    with open(options.interpolated_file_bodypart_1) as fin_StartEnd:
        interpolated_bodypart_coords_1 = dict(json.load(fin_StartEnd))

    with open(options.interpolated_file_bodypart_2) as fin_StartEnd:
        interpolated_bodypart_coords_2 = dict(json.load(fin_StartEnd))


    distMovCoordinateFile = os.path.join('../expts/distance_moving_' + bodypart_1.strip() + "Vs" + bodypart_2.strip()+ ".json")
    fileWriter_distMovCoordinates = open(distMovCoordinateFile, 'w+')

    distance_bodyparts_MH = {}

    for videonumber in range(0, len(interpolated_MH)-1):

        # print videonumber
        distance_bodyparts_MH[videonumber] = {}

        for frameNumber in interpolated_MH[str(videonumber)]["Coordinates"]:

            distance_bodyparts_MH[videonumber][frameNumber] = []
            bodypart_coords_1 = {}
            bodypart_coords_2 = {}
            temp = None

            # print interpolated_bodypart_coords_1[str(videonumber)]["Coordinates"][str(frameNumber)]
            if (frameNumber in interpolated_bodypart_coords_1[str(videonumber)]["Coordinates"]) and (frameNumber in interpolated_bodypart_coords_2[str(videonumber)]["Coordinates"]):

                bodypart_coords_1["x"] = int(interpolated_bodypart_coords_1[str(videonumber)]["Coordinates"][str(frameNumber)]["Interpolated"]["x"])
                bodypart_coords_1["y"] = int(interpolated_bodypart_coords_1[str(videonumber)]["Coordinates"][str(frameNumber)]["Interpolated"]["y"])

                bodypart_coords_2["x"] = int(interpolated_bodypart_coords_2[str(videonumber)]["Coordinates"][str(frameNumber)]["Interpolated"]["x"])
                bodypart_coords_2["y"] = int(interpolated_bodypart_coords_2[str(videonumber)]["Coordinates"][str(frameNumber)]["Interpolated"]["y"])

                # cv2.waitKey(100)
                temp = np.sqrt(np.square(bodypart_coords_2["x"] - bodypart_coords_1["x"]) + np.square(bodypart_coords_2["y"] - bodypart_coords_1["y"])) * 3
                distance_bodyparts_MH[videonumber][frameNumber].append(temp)
                # print temp
            else:
                temp = 0
                distance_bodyparts_MH[videonumber][frameNumber].append(temp)
                # print temp

    json.dump(distance_bodyparts_MH, fileWriter_distMovCoordinates, sort_keys=True, indent=4, separators=(',', ': '))
    fileWriter_distMovCoordinates.close()

    P.figure()

    vidFile_num = 63;
    y = []
    x = []
    print "Video File: ", interpolated_bodypart_coords_1[str(vidFile_num)]["VideoFile"]

    # print distance_bodyparts_MH[vidFile_num]

    for i in distance_bodyparts_MH[vidFile_num]:
        y.append(distance_bodyparts_MH[vidFile_num][str(i)])
        t = int(i)*5
        x.append(t)

    zipped = zip(x, y)
    zipped = sorted(zipped, key=itemgetter(0))
    x, y = zip(*zipped)

    P.plot(x, y, 'bo-')

    P.xlabel('Time (milliseconds)')
    P.ylabel('Distance between Left Dorsal Organ and Right Dorsal Organ (micrometer)')
    P.axis([0, max(x), 0, 200])
    fig_name = '../expts/TimeGraph_LeftDOvsRightDO_distances_' + str(vidFile_num) + '.png'
    P.savefig(fig_name)
    P.show()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--bodypart-1", dest="bodypart_1", default="", help="path containing data directory")
    parser.add_option("", "--bodypart-2", dest="bodypart_2", default="", help="path containing data directory")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--interpolated_file_bodypart_1", dest="interpolated_file_bodypart_1",default="LeftDorsalOrgan", help="Input the bodypart to be trained")
    parser.add_option("", "--interpolated_file_bodypart_2", dest="interpolated_file_bodypart_2",default="RightDorsalOrgan", help="Input the bodypart to be trained")
    parser.add_option("", "--interpolated_file_MH", dest="interpolated_file_MH", help="Input the bodypart to be trained")

    (options, args) = parser.parse_args()

    main(options, args)
