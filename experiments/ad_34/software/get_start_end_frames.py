#! /usr/bin/env python

from optparse import OptionParser
import json
import os
import re
import struct
import cv2
import numpy as np


def main(options, args):

    test_bodypart = options.test_bodypart
    # print "Body Part: ", test_bodypart
    bodypart_coords_gt = {}
    bodypart_coords_interp = {}
    test_annotations = []

    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.append(test_annotation)

    print "len(test_annotations): ", len(test_annotations)

    endStartPositionFile = os.path.join('../expts/end_start_positions_' + test_bodypart.strip() + ".json")
    fileWriter_endStartPosition = open(endStartPositionFile, 'w+')

    for videoNumber in range(0, len(test_annotations)):

        bodypart_coords_gt[videoNumber] = {}
        bodypart_coords_interp[videoNumber] = {}

        bodypart_coords_gt[videoNumber]["StartFrames"] = []
        bodypart_coords_gt[videoNumber]["EndFrames"] = []

        video_file = test_annotations[videoNumber]["VideoFile"]
        video_file = re.sub(".*/data/", "data/", video_file)
        video_file = os.path.join(options.project_dir, video_file)
        print "Video File: ", video_file
        cap = cv2.VideoCapture(video_file)

        bodypart_coords_gt[videoNumber]["VideoFile"] = video_file
        bodypart_coords_interp[videoNumber]["VideoFile"] = video_file

        annotation = []
        annotation.extend(test_annotations[videoNumber]["Annotations"])
        start_f = None
        end_f = None


        for frameNumber in range(0, len(annotation)):
            for j in range(0, len(annotation[frameNumber]["FrameValueCoordinates"])):
                    if (annotation[frameNumber]["FrameValueCoordinates"][j]["Name"] == test_bodypart):
                        bodypart_coords_gt[videoNumber][frameNumber] = {}
                        bodypart_coords_gt[videoNumber][frameNumber]["x"] = int(annotation[frameNumber]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                        bodypart_coords_gt[videoNumber][frameNumber]["y"] = int(annotation[frameNumber]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])
                        bodypart_coords_gt[videoNumber][frameNumber]["FrameIndexVideo"] = int(annotation[frameNumber]["FrameIndexVideo"])
                        bodypart_coords_gt[videoNumber][frameNumber]["BodyPart"] = test_bodypart

        for frameNumber in range(0, len(annotation)):
                for j in range(0, len(annotation[frameNumber]["FrameValueCoordinates"])):
                    if (annotation[frameNumber]["FrameValueCoordinates"][j]["Name"] == test_bodypart):
                        if (annotation[frameNumber]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                            if frameNumber > 0 and frameNumber < len(annotation)-1:
                                if (annotation[frameNumber-1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] == -1)and (annotation[frameNumber+1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                                        start_f = annotation[frameNumber]["FrameIndexVideo"]
                                        bodypart_coords_gt[videoNumber]["StartFrames"].append(start_f)
                                if ((annotation[frameNumber-1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1)and (annotation[frameNumber+1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] == -1)):
                                        end_f = annotation[frameNumber]["FrameIndexVideo"]
                                        bodypart_coords_gt[videoNumber]["EndFrames"].append(end_f)
                            elif frameNumber == 0:
                                if (annotation[frameNumber+1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                                    start_f = annotation[frameNumber]["FrameIndexVideo"]
                                    bodypart_coords_gt[videoNumber]["StartFrames"].append(start_f)
                            elif frameNumber == len(annotation)-1:
                                if (annotation[frameNumber-1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                                        end_f = annotation[frameNumber]["FrameIndexVideo"]
                                        bodypart_coords_gt[videoNumber]["EndFrames"].append(end_f)

    json.dump(bodypart_coords_gt, fileWriter_endStartPosition, sort_keys=True, indent=4, separators=(',', ': '))
    fileWriter_endStartPosition.close()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--bodypart", dest="test_bodypart", default="MouthHook", help="Input the bodypart to be tested")

    (options, args) = parser.parse_args()

    main(options, args)