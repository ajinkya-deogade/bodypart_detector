#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import re
import numpy as np
import pickle
import random
import csv
import os
from tabulate import tabulate

class SaveClass:
    def __init__(self, votes, keypoints, descriptors, bodypart, hessianThreshold, nOctaves, nOctaveLayers):
        self.votes = votes
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bodypart = bodypart
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers

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
    parser.add_option("", "--nOctaves", dest="nOctaves",default="3", help="Input the number of octaves used in surf object")
    parser.add_option("", "--nOctaveLayers", dest="nOctaveLayers",default="3", help="Input the number of octave layers used in surf object")
    parser.add_option("", "--hessian-threshold", dest="hessianThreshold",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--dir-keypoints", dest="dir_keypoints",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--dir-descriptor", dest="dir_descriptors",default="MouthHook", help="Input the bodypart to be trained")

    (options, args) = parser.parse_args()

    if (options.train_annotation_file != ""):
        print "annotation_file:", options.train_annotation_file
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

    surf = cv2.SURF(int(options.hessianThreshold), nOctaves=int(options.nOctaves), nOctaveLayers=int(options.nOctaveLayers))

    training_bodypart = options.train_bodypart
    crop_size = 256
    keypoints_all = []
    count_dist = 0

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
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
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook" and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt = {}
                bodypart_coords_gt["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords_gt["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        # print "Length of Body Coords: ",len(bodypart_coords)
        if (bodypart_coords is not None):
            # try:
                crop_x = max(0, bodypart_coords_gt["x"]-100)
                crop_y = max(0, bodypart_coords_gt["y"]-100)
                frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, 0]

                ## Read Keypoints
                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)
                keypoints_folder = os.path.join(options.dir_keypoints, parent_dir)

                if not os.path.exists(keypoints_folder):
                    print "Folder does not exist !!!"

                keypoints_file = os.path.join(keypoints_folder, os.path.splitext(os.path.basename(train_annotation["Annotations"][i]["FrameFile"]))[0]) + ".csv"

                keypoints = []
                with open(keypoints_file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    for row in csvreader:
                        row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
                        row = row[0]
                        keypoints_temp = cv2.KeyPoint(x=float(row[0]), y=float(row[1]), _size=float((float(row[2])/float(1.2))*float(9)), _angle=float(row[6]), _response=float(row[3]), _octave=int(row[4]), _class_id=int(row[5]))
                        keypoints.append(keypoints_temp)
                        keypoints_all.append(keypoints_temp)

                ## Read Descriptors
                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)
                descriptors_folder = os.path.join(options.dir_descriptors, parent_dir)

                if not os.path.exists(descriptors_folder):
                    print "Folder does not exist !!!"

                descriptors_file = os.path.join(descriptors_folder, os.path.splitext(os.path.basename(train_annotation["Annotations"][i]["FrameFile"]))[0]) + ".csv"

                descriptors = []
                with open(descriptors_file, 'r') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    for row in csvreader:
                        row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
                        row = row[0]
                        descriptors.append(row)

                bodypart_kp_fpga = keypoints
                bodypart_desc_fpga = descriptors

                bodypart_kp_opencv, bodypart_desc_opencv = surf.compute(frame, keypoints)

                # angles = {}
                # angles['fpga(radians)'] = []
                # angles['opencv(radians)'] = []
                # angles['opencv(degrees)'] = []

                for k in range(0, len(bodypart_kp_opencv)):
                    # angles['fpga(radians)'].append(bodypart_kp_fpga[k].angle)
                    # angles['opencv(radians)'].append(np.pi*float(bodypart_kp_opencv[k].angle)/180.0)
                    # angles['opencv(degrees)'].append(bodypart_kp_opencv[k].angle)

                    keypoints_temp = cv2.KeyPoint(x=float(row[0]), y=float(row[1]), _size=float((float(row[2])/float(1.2))*float(9)), _angle=float(2.0*np.pi - row[6]), _response=float(row[3]), _octave=int(row[4]), _class_id=int(row[5]))

                # print angles
                print tabulate(angles, headers = 'keys', numalign="center")
                cv2.waitKey(10000)

            # except:
            #     continue
