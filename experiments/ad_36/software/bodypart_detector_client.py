#! /usr/bin/env python

from optparse import OptionParser
import json
import os
import re
import cv2
import numpy as np


def main(options, args):
    test_annotations = []
    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.extend(test_annotation["Annotations"])

    print "len(test_annotations):", len(test_annotations)

    frame_index = -1
    crop_margin = 256
    bodypart_gt = {}
    countBad = 0

    for j in range(0, len(test_annotations)):

        frame_index += 1
        annotation = test_annotations[j]

        frame_file_0 = annotation["FrameFile"]
        # print frame_file_0

        # for training done on external hard disk. Please change the source me file to HD2
        # frame_file = re.sub(".*/data/Janelia_Q1_2017/20170318_forTraining/", "CRG_Dropbox/AljoComputer/Matlab/high_res_tracker/Data/20170318_DO_Testing/", frame_file_0)
        # frame_file = os.path.join(options.project_dir, frame_file)

        ## Updated the -1: for training done on macbook. Please change the source me file to GoogleDrive
        frame_file = os.path.join(options.project_dir, frame_file_0[1:])
        # print "Frame File: ", frame_file

        frame = cv2.imread(frame_file)
        # print 'frame: ', np.shape(frame)

        bodypart_coords_gt = {}

        for k in range(0, len(annotation["FrameValueCoordinates"])):
            bi = annotation["FrameValueCoordinates"][k]["Name"]
            if ((bi == "MouthHook" or any(bi == s for s in options.detect_bodypart)) and
                    annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"] != -1 and
                    annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt[bi] = {}
                bodypart_coords_gt[bi]["x"] = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                bodypart_coords_gt[bi]["y"] = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])

        if options.verbosity >= 1:
            os.system('clear')
            # os.system('cls')
            print "frame_index:", frame_index
            print "Bad Count :", countBad

        bodypart_gt[frame_index] = {}
        bodypart_gt[frame_index]["bodypart_coords_gt"] = bodypart_coords_gt
        bodypart_gt[frame_index]["frame_file"] = frame_file

        if options.verbosity >= 2:
            print "bodypart_coords_gt:", bodypart_coords_gt

        try:
            if "MouthHook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
                crop_center_x = bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]
                crop_center_y = bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]
            elif "LeftMHhook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
                crop_center_x = bodypart_gt[frame_index]["bodypart_coords_gt"]["LeftMHhook"]["x"]
                crop_center_y = bodypart_gt[frame_index]["bodypart_coords_gt"]["LeftMHhook"]["y"]
            elif "RightMHhook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
                crop_center_x = bodypart_gt[frame_index]["bodypart_coords_gt"]["RightMHhook"]["x"]
                crop_center_y = bodypart_gt[frame_index]["bodypart_coords_gt"]["RightMHhook"]["y"]
            elif "LeftDorsalOrgan" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
                crop_center_x = bodypart_gt[frame_index]["bodypart_coords_gt"]["LeftDorsalOrgan"]["x"]
                crop_center_y = bodypart_gt[frame_index]["bodypart_coords_gt"]["LeftDorsalOrgan"]["y"]
            elif "RightDorsalOrgan" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
                crop_center_x = bodypart_gt[frame_index]["bodypart_coords_gt"]["RightDorsalOrgan"]["x"]
                crop_center_y = bodypart_gt[frame_index]["bodypart_coords_gt"]["RightDorsalOrgan"]["y"]
            else:
                print '************** Not Found 1 **************'
                countBad += 1
                continue

            crop_x = max(0, crop_center_x - (crop_margin) / 2)
            crop_y = max(0, crop_center_y - (crop_margin) / 2)
            image = frame[crop_y:crop_y + crop_margin, crop_x:crop_x + crop_margin, 0]

            if options.display_level >= 1:
                image_2 = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("voters", image_2)

            if options.save_dir_images != "":
                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)
                save_folder = os.path.join(options.save_dir_images, parent_dir)

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_name = os.path.join(save_folder,
                                         os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".png"
                cv2.imwrite(save_name, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        except:
            countBad += 1
            print '************** Not Found 2 **************'
            continue
    print 'Number of Bad Frames = ', countBad


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list", default="",
                      help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--verbosity", dest="verbosity", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",
                      help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--detect-bodypart", dest="detect_bodypart", default="MouthHook", type="string",
                      help="bodypart to detect [MouthHook]")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="",
                      help="directory to save result visualizations, if at all")

    (options, args) = parser.parse_args()

    print options.detect_bodypart

    main(options, args)
