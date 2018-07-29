#! /opt/local/bin/python

import json
import os
import pickle
import random
import re
from optparse import OptionParser

import cv2
import numpy as np


class SaveClass:
    def __init__(self, votes, keypoints, descriptors, bodypart, hessianThreshold, nOctaves, nOctaveLayers):
        self.votes = votes
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bodypart = bodypart
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers

def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

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
    parser.add_option("", "--training-bodypart", dest="train_bodypart",default="MouthHook", action="callback", type="string", callback=string_split, help="Input the bodypart to be trained")
    parser.add_option("", "--nOctaves", dest="nOctaves",default="3", help="Input the number of octaves used in surf object")
    parser.add_option("", "--nOctaveLayers", dest="nOctaveLayers",default="3", help="Input the number of octave layers used in surf object")
    parser.add_option("", "--hessian-threshold", dest="hessianThreshold",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--dir-keypoints", dest="dir_keypoints",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--dir-descriptor", dest="dir_descriptors",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--pos-neg-equal", dest="pos_neg_equal",default="MouthHook", help="Input the bodypart to be trained")

    (options, args) = parser.parse_args()

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

    surf = cv2.xfeatures2d.SURF_create(int(options.hessianThreshold), nOctaves=int(options.nOctaves), nOctaveLayers=int(options.nOctaveLayers))
    class SaveClass:
        def __init__(self, votes, keypoints, descriptors, bodypart, hessianThreshold, nOctaves, nOctaveLayers):
            self.votes = votes
            self.keypoints = keypoints
            self.descriptors = descriptors
            self.bodypart = bodypart
            self.hessianThreshold = hessianThreshold
            self.nOctaves = nOctaves
            self.nOctaveLayers = nOctaveLayers

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
    crop_size = 256
    keypoints_all = 0

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        frame = cv2.imread(frame_file)

        if (options.display_level >= 2):
            display_frame = frame.copy()
        else:
            display_frame = None

        bodypart_coords = []
        bodypart_coords_gt = []
        bodypart_kp_train_pos_frame = []
        bodypart_desc_train_pos_frame = []
        bodypart_vote_train_pos_frame = []
        bodypart_kp_train_neg_frame = []
        bodypart_desc_train_neg_frame = []
        bodypart_vote_train_neg_frame = []
        for j in range(0, len(train_annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords.append({"x" : int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]),
                                        "y" : int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]),
                                        "bodypart" : train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"]})
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook" and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt = {}
                bodypart_coords_gt["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords_gt["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if (bodypart_coords != [] and bodypart_coords_gt != []):
            crop_x = max(0, bodypart_coords_gt["x"]-(crop_size)/2)
            crop_y = max(0, bodypart_coords_gt["y"]-(crop_size)/2)
            frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, 0]

            bodypart_kp, bodypart_desc = surf.detectAndCompute(frame, None)

            for k in range(0, len(bodypart_kp)):
                keypoints_all = keypoints_all + 1
                x, y = bodypart_kp[k].pt
                x = x + crop_x
                y = y + crop_y
                a = np.pi * bodypart_kp[k].angle / 180.0

                flag_add_to_pos = False
                votes_tmp = []
                for bodypart_id in range(0, len(bodypart_coords)):
                    # if key point is less than certain distance from the bodypart
                    if (np.sqrt(np.square(x - bodypart_coords[bodypart_id]["x"]) + 
                                np.square(y - bodypart_coords[bodypart_id]["y"])) <= options.mh_neighborhood):
                        flag_add_to_pos = True
                        bodypart_dp_pos = np.array([bodypart_coords[bodypart_id]["x"] - x, bodypart_coords[bodypart_id]["y"] - y]).T
                        bodypart_R_pos = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                        bodypart_dp_R_pos = np.dot(bodypart_R_pos, bodypart_dp_pos)
                        votes_tmp.append((bodypart_dp_R_pos, bodypart_coords[bodypart_id]["bodypart"]))

                if ( flag_add_to_pos ):
                    bodypart_kp_train_pos_frame.append(bodypart_kp[k])
                    bodypart_desc_train_pos_frame.append(bodypart_desc[k])
                    bodypart_vote_train_pos_frame.append(votes_tmp)

                flag_add_to_neg = False
                for bodypart_id in range(0, len(bodypart_coords)):
                    # if key point is greater than certain distance from the bodypart
                    if (np.sqrt(np.square(x - bodypart_coords[bodypart_id]["x"]) + 
                                np.square(y - bodypart_coords[bodypart_id]["y"])) >= 1.5*options.mh_neighborhood):
                        flag_add_to_neg = True
                        break

                if (flag_add_to_neg):
                    bodypart_kp_train_neg_frame.append(bodypart_kp[k])
                    bodypart_desc_train_neg_frame.append(bodypart_desc[k])

            # print "Number of Positive Training Samples: ", len(bodypart_kp_train_pos_frame)
            # print "Number of Negative Training Samples: ", len(bodypart_kp_train_neg_frame)
            if options.pos_neg_equal:
                if len(bodypart_kp_train_neg_frame) >= len(bodypart_kp_train_pos_frame) and len(bodypart_kp_train_pos_frame) > 0:
                    factor = float(len(bodypart_kp_train_neg_frame))/float(len(bodypart_kp_train_pos_frame))
                    bodypart_kp_train_neg_frame, bodypart_desc_train_neg_frame = zip(*random.sample(zip(bodypart_kp_train_neg_frame, bodypart_desc_train_neg_frame), int(len(bodypart_kp_train_neg_frame)/factor)))
                    bodypart_kp_train_pos.extend(bodypart_kp_train_pos_frame)
                    bodypart_desc_train_pos.extend(bodypart_desc_train_pos_frame)
                    bodypart_vote_train_pos.extend(bodypart_vote_train_pos_frame)
                    bodypart_kp_train_neg.extend(bodypart_kp_train_neg_frame)
                    bodypart_desc_train_neg.extend(bodypart_desc_train_neg_frame)

                elif len(bodypart_kp_train_pos_frame) > len(bodypart_kp_train_neg_frame) and len(bodypart_kp_train_neg_frame) > 0:
                    factor = float(len(bodypart_kp_train_pos_frame))/float(len(bodypart_kp_train_neg_frame))
                    bodypart_kp_train_pos_frame, bodypart_desc_train_pos_frame, bodypart_vote_train_pos_frame = zip(*random.sample(zip(bodypart_kp_train_pos_frame, bodypart_desc_train_pos_frame, bodypart_vote_train_pos_frame), int(len(bodypart_kp_train_pos_frame)/factor)))
                    bodypart_kp_train_pos.extend(bodypart_kp_train_pos_frame)
                    bodypart_desc_train_pos.extend(bodypart_desc_train_pos_frame)
                    bodypart_vote_train_pos.extend(bodypart_vote_train_pos_frame)
                    bodypart_kp_train_neg.extend(bodypart_kp_train_neg_frame)
                    bodypart_desc_train_neg.extend(bodypart_desc_train_neg_frame)

                else:
                    bodypart_kp_train_pos.extend(bodypart_kp_train_pos_frame)
                    bodypart_desc_train_pos.extend(bodypart_desc_train_pos_frame)
                    bodypart_vote_train_pos.extend(bodypart_vote_train_pos_frame)
            else:
                bodypart_kp_train_pos.extend(bodypart_kp_train_pos_frame)
                bodypart_desc_train_pos.extend(bodypart_desc_train_pos_frame)
                bodypart_vote_train_pos.extend(bodypart_vote_train_pos_frame)
                bodypart_kp_train_neg.extend(bodypart_kp_train_neg_frame)
                bodypart_desc_train_neg.extend(bodypart_desc_train_neg_frame)

            if (display_frame != None):
                cv2.circle(display_frame, (bodypart_coords["x"], bodypart_coords["y"]), 4, (0, 0, 255), thickness=-1)
                cv2.circle(display_frame, (bodypart_coords["x"], bodypart_coords["y"]), options.mh_neighborhood, (0, 255, 255),
                           thickness=3)

                if (display_frame != None):
                    display_frame = cv2.resize(display_frame, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("mouth hook annotation", display_frame)

        os.system('cls')
        print "Training Body Part: ", options.train_bodypart
        print "Percentage Complete: %.2f" %(float(i)/float(len(train_annotation["Annotations"]))*100)

    print "Number of Positive Training Samples: ", len(bodypart_kp_train_pos)
    print "Number of Negative Training Samples: ", len(bodypart_kp_train_neg)
    print "Number of All Keypoints: ", keypoints_all

    # # Make the positive and negative sets equal
    # if len(bodypart_vote_train_neg) >= len(bodypart_vote_train_pos):
    #     factor = float(len(bodypart_vote_train_neg))/float(len(bodypart_vote_train_pos))
    #     bodypart_kp_train_neg, bodypart_desc_train_neg, bodypart_vote_train_neg = zip(*random.sample(zip(bodypart_kp_train_neg, bodypart_desc_train_neg, bodypart_vote_train_neg), int(len(bodypart_vote_train_neg)/factor)))
    # else:
    #     factor = float(len(bodypart_vote_train_pos))/float(len(bodypart_vote_train_neg))
    #     bodypart_kp_train_pos, bodypart_desc_train_pos, bodypart_vote_train_pos = zip(*random.sample(zip(bodypart_kp_train_pos, bodypart_desc_train_pos, bodypart_vote_train_pos), int(len(bodypart_vote_train_pos)/factor)))
    #
    # print "Factor: ", factor
    # print "Number of Positive Training Samples Stored: ", len(bodypart_vote_train_pos)
    # print "Number of Negative Training Samples Stored: ", len(bodypart_vote_train_neg)

    bodypart_desc_train_samples_pos = np.array(bodypart_desc_train_pos)
    bodypart_kp_train_responses_pos = np.arange(len(bodypart_kp_train_pos), dtype=np.float32)
    bodypart_desc_train_samples_neg = np.array(bodypart_desc_train_neg)
    bodypart_kp_train_responses_neg = np.arange(len(bodypart_kp_train_neg), dtype=np.float32)

    PosSaveObject = SaveClass(bodypart_vote_train_pos, bodypart_kp_train_responses_pos, bodypart_desc_train_samples_pos, training_bodypart, int(options.hessianThreshold), int(options.nOctaves), int(options.nOctaveLayers))
    with open(options.train_data_pos, 'wb') as fin_save_pos:
        pickle.dump(PosSaveObject, fin_save_pos)

    NegSaveObject = SaveClass(bodypart_vote_train_neg, bodypart_kp_train_responses_neg, bodypart_desc_train_samples_neg, training_bodypart, int(options.hessianThreshold), int(options.nOctaves), int(options.nOctaveLayers))
    with open(options.train_data_neg, 'wb') as fin_save_neg:
        pickle.dump(NegSaveObject, fin_save_neg)
