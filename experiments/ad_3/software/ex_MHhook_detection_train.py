
#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle

if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list", dest="train_annotation_list", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--mh-neighborhood", dest="mh_neighborhood", type="int", default=10,help="distance from mouth hook for a keyppoint to be considered relevant for training")
    parser.add_option("", "--training-datafile", dest="train_data", help="File to save the information about the data")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--training-bodypart", dest="train_bodypart",default="MouthHook", help="Input the bodypart to be trained")

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
                train_annotation_file = os.path.join(options.project_dir,
                                                     re.sub(".*/data/", "data/", train_annotation_file.strip()))
                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    train_annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    # Histogram Equalisation Object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # SURF Detector Object
    surf = cv2.SURF(400, nOctaves=4, nOctaveLayers=4)
    class SaveClass:
        def __init__(self, votes, keypoints, descriptors, bodypart):
            self.votes = votes
            self.keypoints = keypoints
            self.descriptors = descriptors
            self.bodypart = bodypart

    rmh_kp_train = []
    rmh_desc_train = []
    rmh_vote_train = []
    training_bodypart = options.train_bodypart

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        #print frame_file
        # Read Frame
        frame = cv2.imread(frame_file)

        if (options.display_level >= 2):
            display_frame = frame.copy()
        else:
            display_frame = None

        rmh_coords = None
        for j in range(0, len(train_annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                rmh_coords = {}
                rmh_coords["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                rmh_coords["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if (rmh_coords is not None):
            rmh_kp, rmh_desc = surf.detectAndCompute(frame, None)
            for k in range(0, len(rmh_kp)):
                x, y = rmh_kp[k].pt
                a = np.pi * rmh_kp[k].angle / 180.0
                # if key point is less than certain distance from the
                if (np.sqrt(np.square(x - rmh_coords["x"]) + np.square(
                            y - rmh_coords["y"])) <= options.mh_neighborhood):
                    rmh_kp_train.append(rmh_kp[k])
                    rmh_desc_train.append(rmh_desc[k])
                    rmh_dp = np.array([rmh_coords["x"] - x, rmh_coords["y"] - y]).T
                    rmh_R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                    rmh_dp_R = np.dot(rmh_R, rmh_dp)
                    rmh_vote_train.append(rmh_dp_R)

            if (display_frame != None):
                cv2.circle(display_frame, (rmh_coords["x"], rmh_coords["y"]), 4, (0, 0, 255), thickness=-1)
                cv2.circle(display_frame, (rmh_coords["x"], rmh_coords["y"]), options.mh_neighborhood, (0, 255, 255),
                           thickness=3)

        if (display_frame != None):
            display_frame = cv2.resize(display_frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("mouth hook annotation", display_frame)
        rmh_desc_train_samples = np.array(rmh_desc_train)
        rmh_kp_train_responses = np.arange(len(rmh_kp_train), dtype=np.float32)

    SaveObject = SaveClass(rmh_vote_train, rmh_kp_train_responses, rmh_desc_train_samples, training_bodypart)
    with open(options.train_data, 'wb') as fin_save:
        pickle.dump(SaveObject, fin_save)
