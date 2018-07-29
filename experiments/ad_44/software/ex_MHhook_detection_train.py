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
import csv
import os
class SaveClass:
    def __init__(self, votes, keypoints, descriptors, bodypart):
        self.votes = votes
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bodypart = bodypart

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
    parser.add_option("", "--keypoint-dir", dest="dir_keypoints",default="MouthHook", help="Input the bodypart to be trained")

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

    # print "Len Annotaions: ", len(train_annotation["Annotations"])
    surf = cv2.SURF(500, nOctaves=2, nOctaveLayers=3)
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
    print "Bodypart:", training_bodypart
    crop_size = 256

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        # print frame_file
        # Read Frame
        frame = cv2.imread(frame_file)


        if (options.display_level >= 2):
            display_frame = frame.copy()
        else:
            display_frame = None

        bodypart_coords = None
        for j in range(0, len(train_annotation["Annotations"][i]["FrameValueCoordinates"])):
            # print train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"]
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                # print "Found a Match !!"
                bodypart_coords = {}
                bodypart_coords["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook" and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt = {}
                bodypart_coords_gt["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords_gt["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if (bodypart_coords is not None):
            # print "bodypart_coords is not None"
            try:
                crop_x = max(0, bodypart_coords_gt["x"]-100)
                crop_y = max(0, bodypart_coords_gt["y"]-100)
                frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, 0]

                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)
                csv_folder = os.path.join(options.dir_keypoints, parent_dir)

                if not os.path.exists(csv_folder):
                    print "Folder does not exist !!!"

                csv_file = os.path.join(csv_folder, os.path.splitext(os.path.basename(train_annotation["Annotations"][i]["FrameFile"]))[0]) + ".csv"

                keypoints = []
                with open(csv_file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    for row in csvreader:
                        row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
                        row = row[0]
                        keypoints_temp = cv2.KeyPoint(x=float(row[0]), y=float(row[1]), _size=float((float(row[2])/float(1.2))*float(9)), _angle=float(0), _response=float(row[3]), _octave=int(row[4]), _class_id=int(row[5]))
                        keypoints.append(keypoints_temp)

                bodypart_kp, bodypart_desc = surf.compute(frame, keypoints)

                for k in range(0, len(bodypart_kp)):
                    x, y = bodypart_kp[k].pt
                    x = x + crop_x
                    y = y + crop_y
                    a = np.pi * bodypart_kp[k].angle / 180.0
                    dist_kp_bp = np.sqrt(np.square(x - bodypart_coords["x"]) + np.square(y - bodypart_coords["y"]))
                    # print "Distance KP: ", dist_kp_bp
                    if (dist_kp_bp <= options.mh_neighborhood):
                        bodypart_kp_train_pos.append(bodypart_kp[k])
                        bodypart_desc_train_pos.append(bodypart_desc[k])
                        bodypart_dp_pos = np.array([bodypart_coords["x"] - x, bodypart_coords["y"] - y]).T
                        bodypart_R_pos = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                        bodypart_dp_R_pos = np.dot(bodypart_R_pos, bodypart_dp_pos)
                        bodypart_vote_train_pos.append(bodypart_dp_R_pos)

                    if (dist_kp_bp >= 1.2*options.mh_neighborhood):
                        bodypart_kp_train.append(bodypart_kp[k])
                        bodypart_desc_train.append(bodypart_desc[k])
                        bodypart_dp = np.array([bodypart_coords["x"] - x, bodypart_coords["y"] - y]).T
                        bodypart_R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                        bodypart_dp_R = np.dot(bodypart_R, bodypart_dp)
                        bodypart_vote_train.append(bodypart_dp_R)
            except:
                continue

        bodypart_kp_train_neg, bodypart_desc_train_neg, bodypart_vote_train_neg = zip(*random.sample(zip(bodypart_kp_train, bodypart_desc_train, bodypart_vote_train), len(bodypart_vote_train)))

        os.system('clear')
        print "Percentage Complete: %.2f" %(float(i)/float(len(train_annotation["Annotations"]))*100)

    print "Positive Keypoints Added : ", len(bodypart_kp_train_pos)
    print "Negative Keypoints Added : ", len(bodypart_kp_train_neg)

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