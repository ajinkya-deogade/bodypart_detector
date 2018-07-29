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

surf = cv2.SURF(500, nOctaves=2, nOctaveLayers=3)

def main(options, args):
    if (options.image_file != ""):
        image = cv2.imread(options.image_file)
        if (options.save_dir_keypoints != ""):
            kp_frame, desc_frame = surf.detectAndCompute(image, None)
            keypoints = []
            descriptors = []

            save_folder_keypoints = options.save_dir_keypoints
            save_folder_descriptors = options.save_dir_descriptors
            save_folder_images = options.save_dir_images

            if not os.path.exists(save_folder_keypoints):
                os.makedirs(save_folder_keypoints)
            if not os.path.exists(save_folder_descriptors):
                os.makedirs(save_folder_descriptors)
            if not os.path.exists(save_folder_images):
                os.makedirs(save_folder_images)

            save_name_keypoints = os.path.join(save_folder_keypoints, os.path.basename(options.image_file)) + ".csv"
            save_name_descriptors = os.path.join(save_folder_descriptors, os.path.basename(options.image_file)) + ".csv"
            save_name_images = os.path.join(save_folder_images, os.path.basename(options.image_file)) + ".tiff"

            for kp in range(0, len(kp_frame)):
                keypoints.append([])
                descriptors.append([])

            for kp_n2 in range(0,len(keypoints)):
                keypoints[kp_n2].append(kp_frame[kp_n2].pt[0])
                keypoints[kp_n2].append(kp_frame[kp_n2].pt[1])
                keypoints[kp_n2].append(kp_frame[kp_n2].size)
                keypoints[kp_n2].append(kp_frame[kp_n2].angle)
                keypoints[kp_n2].append(kp_frame[kp_n2].response)
                keypoints[kp_n2].append(kp_frame[kp_n2].octave)
                keypoints[kp_n2].append(round(kp_frame[kp_n2].class_id))
                descriptors[kp_n2].append(desc_frame[kp_n2])

            with open(save_name_keypoints, 'w') as fp:
                a = csv.writer(fp, delimiter=',')
                a.writerows(keypoints)

            with open(save_name_descriptors, 'w') as fp:
                a = csv.writer(fp, delimiter=',')
                a.writerows(descriptors)

            cv2.imwrite(save_name_images, image, (cv2.cv.CV_IMWRITE_JPEG_QUALITY,50))
    else:
        if (options.annotation_file != ""):
            print "annotation_file:", options.annotation_file
            with open(options.annotation_file) as fin_annotation:
                annotation = json.load(fin_annotation)
        else:
            annotation = {}
            annotation["Annotations"] = []
            with open(options.annotation_list) as fin_annotation_list:
                for train_annotation_file in fin_annotation_list:
                    train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                    with open(train_annotation_file) as fin_annotation:
                        tmp_train_annotation = json.load(fin_annotation)
                        annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    training_bodypart = options.train_bodypart
    crop_size = 256
    for i in range(0, len(annotation["Annotations"])):
        frame_file = annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        frame = cv2.imread(frame_file)
        print "Frame Number: ", i
        os.system('cls')

        bodypart_coords = None
        for j in range(0, len(annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart and annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords = {}
                bodypart_coords["x"] = int(annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords["y"] = int(annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])
            if (annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook" and annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt = {}
                bodypart_coords_gt["x"] = int(annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords_gt["y"] = int(annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if (bodypart_coords is not None):
            crop_x = max(0, bodypart_coords_gt["x"]-100)
            crop_y = max(0, bodypart_coords_gt["y"]-100)
            frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, 0]

            if (options.save_dir_keypoints != ""):
                kp_frame, desc_frame = surf.detectAndCompute(frame, None)
                keypoints = []
                descriptors = []
                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)

                for kp in range(0, len(kp_frame)):
                    keypoints.append([])
                    descriptors.append([])

                for kp_n2 in range(0,len(keypoints)):
                    keypoints[kp_n2].append(kp_frame[kp_n2].pt[0])
                    keypoints[kp_n2].append(kp_frame[kp_n2].pt[1])
                    keypoints[kp_n2].append((float(kp_frame[kp_n2].size)*float(1.2))/float(9))
                    # keypoints[kp_n2].append(kp_frame[kp_n2].angle)
                    keypoints[kp_n2].append(kp_frame[kp_n2].response)
                    keypoints[kp_n2].append(kp_frame[kp_n2].octave)
                    keypoints[kp_n2].append(round(kp_frame[kp_n2].class_id))
                    descriptors[kp_n2].append(desc_frame[kp_n2])

                if (options.save_dir_keypoints != ""):
                    save_folder_keypoints = os.path.join(options.save_dir_keypoints, parent_dir)
                    if not os.path.exists(save_folder_keypoints):
                        os.makedirs(save_folder_keypoints)
                    save_name_keypoints = os.path.join(save_folder_keypoints, os.path.splitext(os.path.basename(annotation["Annotations"][i]["FrameFile"]))[0]) + ".csv"
                    with open(save_name_keypoints, 'w') as fp:
                        a = csv.writer(fp, delimiter=',')
                        a.writerows(keypoints)

                if (options.save_dir_descriptors != ""):
                    save_folder_descriptors = os.path.join(options.save_dir_descriptors, parent_dir)
                    if not os.path.exists(save_folder_descriptors):
                        os.makedirs(save_folder_descriptors)
                    save_name_descriptors = os.path.join(save_folder_descriptors, os.path.splitext(os.path.basename(annotation["Annotations"][i]["FrameFile"]))[0]) + ".csv"
                    with open(save_name_descriptors, 'w') as fp:
                        a = csv.writer(fp, delimiter=',')
                        a.writerows(descriptors)

                if (options.save_dir_images != ""):
                    save_folder_images = os.path.join(options.save_dir_images, parent_dir)
                    if not os.path.exists(save_folder_images):
                        os.makedirs(save_folder_images)
                    save_name_images = os.path.join(save_folder_images, os.path.splitext(os.path.basename(annotation["Annotations"][i]["FrameFile"]))[0]) + ".tiff"
                    cv2.imwrite(save_name_images, frame, (cv2.cv.CV_IMWRITE_JPEG_QUALITY,50))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--annotation-file", dest="annotation_file", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--annotation-list", dest="annotation_list", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--training-bodypart", dest="train_bodypart",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--verbosity", dest="verbosity", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--detect-bodypart", dest="detect_bodypart", default="MouthHook", type="string", help="bodypart to detect [MouthHook]")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--save-dir-keypoints", dest="save_dir_keypoints", default="", help="directory to save keypoints")
    parser.add_option("", "--save-dir-descriptors", dest="save_dir_descriptors", default="", help="directory to save keypoints")
    parser.add_option("", "--image-file", dest="image_file", default="", help="directory to save keypoints")

    (options, args) = parser.parse_args()

    print options.detect_bodypart

    main(options, args)