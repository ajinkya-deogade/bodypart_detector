#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle
import random
import pylab as P
# from scipy import stats
# import matplotlib.mlab as mlab


if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--interpolated-coordinate-file", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list", dest="train_annotation_list", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--training-bodypart-1", dest="train_bodypart_1",default="LeftDorsalOrgan", help="Input the bodypart to be trained")
    parser.add_option("", "--training-bodypart-2", dest="train_bodypart_2",default="RightDorsalOrgan", help="Input the bodypart to be trained")


    (options, args) = parser.parse_args()

    print "annotation_file:", options.train_annotation_file

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            train_annotation = json.load(fin_annotation)
    else:
        train_annotation = {}
        train_annotation["Annotations"] = []
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    train_annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    print "Train Annotations: ", len(train_annotation["Annotations"])
    training_bodypart_1 = options.train_bodypart_1
    # print "AS", training_bodypart_1
    training_bodypart_2 = options.train_bodypart_2
    distance_bodyparts_total = []
    mean_distance_bodyparts_total = None
    std_distance_bodyparts_total = None

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        # print "Frame File :", frame_file
        # frame = cv2.imread(frame_file)
        print "Frame: ", i

        bodypart_coords_1 = None
        bodypart_coords_2 = None

        for j in range(0, len(train_annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart_1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords_1 = {}
                bodypart_coords_1["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords_1["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart_2 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords_2 = {}
                bodypart_coords_2["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords_2["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if (bodypart_coords_1 is not None and bodypart_coords_2 is not None):
            temp = np.sqrt(np.square(bodypart_coords_2["x"] - bodypart_coords_1["x"]) + np.square(bodypart_coords_2["y"] - bodypart_coords_1["y"]))
            distance_bodyparts_total.append(temp)

    distance_bodyparts_total[:] = [float(x) * float(3.00) for x in distance_bodyparts_total]
    group_1 = [ x for x in distance_bodyparts_total if (x<=110)]
    group_2 = [ x for x in distance_bodyparts_total if (x>=110)]

    bins = np.linspace(0, 300, 30)
    mean_distance_bodyparts_total = np.mean(distance_bodyparts_total)
    std_distance_bodyparts_total = np.std(distance_bodyparts_total)

    P.figure()
    n, bins, patches = P.hist(distance_bodyparts_total, bins, normed = 1, histtype='bar', rwidth=0.8, color='b')
    # P.axis([0, 250, 0, 0.020])
    P.xlabel('Distance between Left MH-hook and Right Dorsal Organ (micrometer)')
    P.ylabel('Proportion')

    # for pat in patches[0:11]:
    #     P.setp(pat, 'facecolor', 'r', alpha = 0.5)
    # P.axvline(bins[6], color='y', linestyle='dashed', linewidth=2)
    #
    # for pat in patches[11:30]:
    #     P.setp(pat, 'facecolor', 'g', alpha = 0.5)
    # P.axvline(bins[16]- 5, color='b', linestyle='dashed', linewidth=2)

    P.savefig('../expts/Complete_Histogram_LeftMHhookvsRightDO_distances_NO_groups.png')
    P.show()

    with open('../expts/distance_bodyparts_total.txt', 'w') as outfile:
        json.dump(distance_bodyparts_total, outfile)
    with open('../expts/mean_distance_bodyparts_total.txt', 'w') as outfile:
        json.dump(mean_distance_bodyparts_total, outfile)
    with open('../expts/std_distance_bodyparts_total.txt', 'w') as outfile:
        json.dump(std_distance_bodyparts_total, outfile)