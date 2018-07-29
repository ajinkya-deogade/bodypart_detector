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

    for j in range(0, len(test_annotations)):

        frame_index += 1
        annotation = test_annotations[j]

        frame_file_0 = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file_0)
        frame_file = os.path.join(options.project_dir, frame_file)
        frame = cv2.imread(frame_file)
        bodypart_coords_gt = {}

        for k in range(0, len(annotation["FrameValueCoordinates"])):
            bi = annotation["FrameValueCoordinates"][k]["Name"]
            if ((bi == "MouthHook" or any(bi == s for s in options.detect_bodypart)) and annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt[bi] = {}
                bodypart_coords_gt[bi]["x"] = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                bodypart_coords_gt[bi]["y"] = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])

        if ( options.verbosity >= 1 ):
            print "frame_index:", frame_index

        bodypart_gt[frame_index] = {}
        bodypart_gt[frame_index]["bodypart_coords_gt"] = bodypart_coords_gt
        bodypart_gt[frame_index]["frame_file"] = frame_file

        image = copy.deepcopy(frame)

        if ( options.verbosity >= 1 ):
            print "bodypart_coords_gt:" , bodypart_coords_gt

        try:
        #     print "Body Part", bodypart_gt[frame_index]["bodypart_coords_gt"]
        #     if bodypart_gt[frame_index]["bodypart_coords_gt"] == "MouthHook":
            crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-100)
            crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-100)

            image = image[crop_y:crop_y+crop_margin,crop_x:crop_x+crop_margin,0]

            if (options.display_level >= 1):
                image_2 = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("voters", image_2)

            if (options.save_dir_keypoints != ""):
                kp_frame, desc_frame = surf.detectAndCompute(image, None)
                kp_n = 0
                keypoints = []
                descriptors = []

                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)

                save_folder_keypoints = os.path.join(options.save_dir_keypoints, parent_dir)
                save_folder_descriptors = os.path.join(options.save_dir_descriptors, parent_dir)
                save_folder_images = os.path.join(options.save_dir_images, parent_dir)

                if not os.path.exists(save_folder_keypoints):
                    os.makedirs(save_folder_keypoints)
                if not os.path.exists(save_folder_descriptors):
                    os.makedirs(save_folder_descriptors)
                if not os.path.exists(save_folder_images):
                    os.makedirs(save_folder_images)

                save_name_keypoints = os.path.join(save_folder_keypoints, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"
                save_name_descriptors = os.path.join(save_folder_descriptors, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"
                save_name_images = os.path.join(save_folder_images, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".tiff"

                for kp in range(0,len(kp_frame)):
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
                crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-100)
                crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-100)
                image = image[crop_y:crop_y+crop_margin,crop_x:crop_x+crop_margin,0]

                if (options.display_level >= 1):
                    image_2 = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("voters", image_2)

                if (options.save_dir_keypoints != ""):
                    kp_frame, desc_frame = surf.detectAndCompute(image, None)
                    kp_n = 0
                    keypoints = []
                    descriptors = []

                    current_dir = os.path.abspath(os.path.dirname(frame_file))
                    parent_dir = os.path.basename(current_dir)

                    save_folder_keypoints = os.path.join(options.save_dir_keypoints, parent_dir)
                    save_folder_descriptors = os.path.join(options.save_dir_descriptors, parent_dir)
                    save_folder_images = os.path.join(options.save_dir_images, parent_dir)

                    if not os.path.exists(save_folder_keypoints):
                        os.makedirs(save_folder_keypoints)
                    if not os.path.exists(save_folder_descriptors):
                        os.makedirs(save_folder_descriptors)
                    if not os.path.exists(save_folder_images):
                        os.makedirs(save_folder_images)

                    save_name_keypoints = os.path.join(save_folder_keypoints, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"
                    save_name_descriptors = os.path.join(save_folder_descriptors, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"
                    save_name_images = os.path.join(save_folder_images, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".tiff"

                    for kp in range(0,len(kp_frame)):
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
        except:
            continue

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--verbosity", dest="verbosity", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--detect-bodypart", dest="detect_bodypart", default="MouthHook", type="string", help="bodypart to detect [MouthHook]")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--save-dir-keypoints", dest="save_dir_keypoints", default="", help="directory to save keypoints")
    parser.add_option("", "--save-dir-descriptors", dest="save_dir_descriptors", default="", help="directory to save keypoints")

    (options, args) = parser.parse_args()

    print options.detect_bodypart

    main(options, args)
