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

if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list", dest="train_annotation_list", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--mh-neighborhood", dest="mh_neighborhood", type="int", default=10,help="distance from mouth hook for a keyppoint to be considered relevant for training")
    parser.add_option("", "--positive-training-datafile", dest="train_data_pos", help="File to save the information about the positive training data")
    parser.add_option("", "--negative-training-datafile", dest="train_data_neg", help="File to save the information about the negative training data")
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
                train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    train_annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # SURF Detector Object
    surf = cv2.SURF(400, nOctaves=4, nOctaveLayers=4)
    class SaveClass:
        def __init__(self, votes, keypoints, descriptors, bodypart):
            self.votes = votes
            self.keypoints = keypoints
            self.descriptors = descriptors
            self.bodypart = bodypart

    bodypart_kp_train = []
    bodypart_desc_train = []
    bodypart_vote_train = []
    bodypart_kp_train_pos = []
    bodypart_desc_train_pos = []
    bodypart_vote_train_pos = []
    bodypart_kp_train_neg = []
    bodypart_desc_train_neg = []
    bodypart_vote_train_neg = []
    training_bodypart = options.train_bodypart

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)

        print frame_file

        # Read Frame
        frame = cv2.imread(frame_file)

        if (options.display_level >= 2):
            display_frame = frame.copy()
        else:
            display_frame = None

        bodypart_coords = None
        for j in range(0, len(train_annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords = {}
                bodypart_coords["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if (bodypart_coords is not None):
            bodypart_kp, bodypart_desc = surf.detectAndCompute(frame, None)
            for k in range(0, len(bodypart_kp)):
                x, y = bodypart_kp[k].pt
                a = np.pi * bodypart_kp[k].angle / 180.0
                # if key point is less than certain distance from the
                if (np.sqrt(np.square(x - bodypart_coords["x"]) + np.square(
                            y - bodypart_coords["y"])) <= options.mh_neighborhood):
                    bodypart_kp_train_pos.append(bodypart_kp[k])
                    bodypart_desc_train_pos.append(bodypart_desc[k])
                    bodypart_dp_pos = np.array([bodypart_coords["x"] - x, bodypart_coords["y"] - y]).T
                    bodypart_R_pos = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                    bodypart_dp_R_pos = np.dot(bodypart_R_pos, bodypart_dp_pos)
                    bodypart_vote_train_pos.append(bodypart_dp_R_pos)

                if (np.sqrt(np.square(x - bodypart_coords["x"]) + np.square(
                            y - bodypart_coords["y"])) >= 2*options.mh_neighborhood):
                    bodypart_kp_train.append(bodypart_kp[k])
                    bodypart_desc_train.append(bodypart_desc[k])
                    bodypart_dp = np.array([bodypart_coords["x"] - x, bodypart_coords["y"] - y]).T
                    bodypart_R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                    bodypart_dp_R = np.dot(bodypart_R, bodypart_dp)
                    bodypart_vote_train.append(bodypart_dp_R)

            bodypart_kp_train_neg, bodypart_desc_train_neg, bodypart_vote_train_neg = zip(*random.sample(zip(bodypart_kp_train,bodypart_desc_train, bodypart_vote_train),len(bodypart_vote_train)/20))

            if (display_frame != None):
                cv2.circle(display_frame, (bodypart_coords["x"], bodypart_coords["y"]), 4, (0, 0, 255), thickness=-1)
                cv2.circle(display_frame, (bodypart_coords["x"], bodypart_coords["y"]), options.mh_neighborhood, (0, 255, 255),
                           thickness=3)

        if (display_frame != None):
            display_frame = cv2.resize(display_frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("mouth hook annotation", display_frame)

        print "Frame Number: ", i
        print np.shape(bodypart_desc_train_pos)[0]
        print np.shape(bodypart_desc_train_neg)[0]

        bodypart_desc_train_samples_pos = np.array(bodypart_desc_train_pos)
        bodypart_kp_train_responses_pos = np.arange(len(bodypart_kp_train_pos), dtype=np.float32)
        bodypart_desc_train_samples_neg = np.array(bodypart_desc_train_neg)
        bodypart_kp_train_responses_neg = np.arange(len(bodypart_kp_train_neg), dtype=np.float32)

    PosSaveObject = SaveClass(bodypart_vote_train_pos, bodypart_kp_train_responses_pos, bodypart_desc_train_samples_pos, training_bodypart)
    with open(options.train_data_pos, 'wb') as fin_save_pos:
        pickle.dump(PosSaveObject, fin_save_pos)

    NegSaveObject = SaveClass(bodypart_vote_train_neg, bodypart_kp_train_responses_neg, bodypart_desc_train_samples_neg, training_bodypart)
    with open(options.train_data_neg, 'wb') as fin_save_neg:
        pickle.dump(NegSaveObject, fin_save_neg)
