#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import os
import re
import struct
import cv2
import numpy as np
import socket
import sys
import copy
import csv
import glob

surf = cv2.SURF(500, nOctaves=2, nOctaveLayers=3)
surf_descriptorExtractor = cv2.DescriptorExtractor_create("SURF")

def main(options, args):

    test_annotations = []
    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.extend(test_annotation["Annotations"])

    print "len(test_annotations):", len(test_annotations)

    frame_index = -1
    crop_margin = 256
    bodypart_gt = {}
    keypoints = []
    descriptors = np.empty([1, 128], dtype=None)

    for j in range(0, len(test_annotations)):

        frame_index += 1
        annotation = test_annotations[j]

        frame_file_0 = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file_0)
        frame_file = os.path.join(options.project_dir, frame_file)

        # frame = cv2.imread(frame_file)
        # bodypart_coords_gt = {}
        #
        # for k in range(0, len(annotation["FrameValueCoordinates"])):
        #     bi = annotation["FrameValueCoordinates"][k]["Name"]
        #     if ((bi == "MouthHook" or any(bi == s for s in options.detect_bodypart)) and annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"] != -1):
        #         bodypart_coords_gt[bi] = {}
        #         bodypart_coords_gt[bi]["x"] = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
        #         bodypart_coords_gt[bi]["y"] = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])
        #
        # if ( options.verbosity >= 1 ):
        #     print "frame_index:", frame_index
        #
        # bodypart_gt[frame_index] = {}
        # bodypart_gt[frame_index]["bodypart_coords_gt"] = bodypart_coords_gt
        # bodypart_gt[frame_index]["frame_file"] = frame_file
        #
        # image = copy.deepcopy(frame)
        #
        # if ( options.verbosity >= 1 ):
        #     print "bodypart_coords_gt:" , bodypart_coords_gt
        #
        # try:
        #     crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-100)
        #     crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-100)
        #     image = image[crop_y:crop_y+crop_margin,crop_x:crop_x+crop_margin,0]
        # print j
        # keypoints[j] = []
        # descriptors[j] = []

        current_dir = os.path.abspath(os.path.dirname(frame_file))
        parent_dir = os.path.basename(current_dir)
        csv_folder = os.path.join(options.dir_keypoints, parent_dir)
        image_folder = os.path.join(options.dir_images, parent_dir)

        if not os.path.exists(csv_folder):
            print "Folder does not exist !!!"

        if not os.path.exists(image_folder):
            print "Folder does not exist !!!"

        csv_file = os.path.join(csv_folder, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"
        image_file = os.path.join(image_folder, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".jpeg"
        keypoints.append([])

        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            img = cv2.imread(image_file, 0)
            for row in csvreader:
                row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
                row = row[0]
                temp_feature = cv2.KeyPoint(x=float(row[0]), y=float(row[1]), _size=float((float(row[2])/float(1.2))*float(9)), _angle=float(-1), _response=float(row[3]), _octave=int(row[4]), _class_id=int(row[5]))
                keypoints[j].append(temp_feature)

        print np.shape(keypoints[j])
        kp, desc = surf_descriptorExtractor.compute(img, keypoints[j])

        print np.shape(desc)
        cv2.waitKey(1000)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--verbosity", dest="verbosity", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--detect-bodypart", dest="detect_bodypart", default="MouthHook", type="string", help="bodypart to detect [MouthHook]")
    parser.add_option("", "--dir-images", dest="dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--dir-keypoints", dest="dir_keypoints", default="", help="directory to save keypoints")

    (options, args) = parser.parse_args()

    print options.detect_bodypart

    main(options, args)
