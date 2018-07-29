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

from multiprocessing import Pool
from multiprocessing import Manager
from threading import Lock

# Histogram Equalisation Object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# SURF Detector Object
surf = cv2.SURF(500, nOctaves=2, nOctaveLayers=2)

class SaveClass:
    def __init__(self, votes, keypoints, descriptors, bodypart):
        self.votes = votes
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bodypart = bodypart

lock = Lock()
manager = Manager()
bodypart_desc_train_pos_all = manager.list([])
bodypart_vote_train_pos_all = manager.list([])
bodypart_desc_train_neg_all = manager.list([])
bodypart_vote_train_neg_all = manager.list([])

options = None
args = None

def get_train_features(annotation):
    global bodypart_desc_train_pos_all, bodypart_vote_train_pos_all, bodypart_desc_train_neg_all, bodypart_vote_train_neg_all
    global options, args

    bodypart_desc_train = []
    bodypart_vote_train = []
    bodypart_desc_train_pos = []
    bodypart_vote_train_pos = []
    bodypart_desc_train_neg = []
    bodypart_vote_train_neg = []
    training_bodypart = options.train_bodypart

    frame_file = annotation["FrameFile"]
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
    for j in range(0, len(annotation["FrameValueCoordinates"])):
        if (annotation["FrameValueCoordinates"][j]["Name"] == training_bodypart and 
            annotation["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and 
            annotation["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
            bodypart_coords = {}
            bodypart_coords["x"] = int(annotation["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
            bodypart_coords["y"] = int(annotation["FrameValueCoordinates"][j]["Value"]["y_coordinate"])


    if (bodypart_coords is not None):
        bodypart_kp, bodypart_desc = surf.detectAndCompute(frame, None)
        for k in range(0, len(bodypart_kp)):
            x, y = bodypart_kp[k].pt
            a = np.pi * bodypart_kp[k].angle / 180.0
            # if key point is less than certain distance from the
            if (np.sqrt(np.square(x - bodypart_coords["x"]) + np.square(
                    y - bodypart_coords["y"])) <= options.mh_neighborhood):
                bodypart_desc_train_pos.append(bodypart_desc[k])
                bodypart_dp_pos = np.array([bodypart_coords["x"] - x, bodypart_coords["y"] - y]).T
                bodypart_R_pos = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                bodypart_dp_R_pos = np.dot(bodypart_R_pos, bodypart_dp_pos)
                bodypart_vote_train_pos.append(bodypart_dp_R_pos)

            if (np.sqrt(np.square(x - bodypart_coords["x"]) + np.square(
                    y - bodypart_coords["y"])) >= 2*options.mh_neighborhood):
                bodypart_desc_train.append(bodypart_desc[k])
                bodypart_dp = np.array([bodypart_coords["x"] - x, bodypart_coords["y"] - y]).T
                bodypart_R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                bodypart_dp_R = np.dot(bodypart_R, bodypart_dp)
                bodypart_vote_train.append(bodypart_dp_R)

        bodypart_desc_train_neg, bodypart_vote_train_neg = zip(*random.sample(zip(bodypart_desc_train, bodypart_vote_train),len(bodypart_vote_train_pos)))

        if (display_frame != None):
            cv2.circle(display_frame, (bodypart_coords["x"], bodypart_coords["y"]), 4, (0, 0, 255), thickness=-1)
            cv2.circle(display_frame, (bodypart_coords["x"], bodypart_coords["y"]), options.mh_neighborhood, (0, 255, 255),
                       thickness=3)

    if (display_frame != None):
        display_frame = cv2.resize(display_frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("mouth hook annotation", display_frame)

    lock.acquire()
    bodypart_desc_train_pos_all.extend(bodypart_desc_train_pos)
    bodypart_vote_train_pos_all.extend(bodypart_vote_train_pos)
    bodypart_desc_train_neg_all.extend(bodypart_desc_train_neg)
    bodypart_vote_train_neg_all.extend(bodypart_vote_train_neg)
    lock.release()

    return True


def main(options, args):
    global bodypart_desc_train_pos_all, bodypart_vote_train_pos_all, bodypart_desc_train_neg_all, bodypart_vote_train_neg_all, keypoints, descriptors
    keypoints = []
    descriptors = np.empty([1, 128], dtype=None)

    training_bodypart = options.train_bodypart

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            print "annotation_file:", options.train_annotation_file
            train_annotation = json.load(fin_annotation)
    else:
        train_annotation = {}
        train_annotation["Annotations"] = []
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                print "annotation_file:", train_annotation_file
                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    train_annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    process_pool = Pool(processes=options.n_thread)
    results = process_pool.map(get_train_features, train_annotation["Annotations"])
    process_pool.close()
    process_pool.join()

    lock.acquire()
    bodypart_vote_train_pos_all_ = []
    for v in bodypart_vote_train_pos_all:
        bodypart_vote_train_pos_all_.append(v)
    bodypart_desc_train_samples_pos = np.array(bodypart_desc_train_pos_all)
    bodypart_kp_train_responses_pos = np.arange(len(bodypart_desc_train_pos_all), dtype=np.float32)
    bodypart_vote_train_neg_all_ = []
    for v in bodypart_vote_train_neg_all:
        bodypart_vote_train_neg_all_.append(v)
    bodypart_desc_train_samples_neg = np.array(bodypart_desc_train_neg_all)
    bodypart_kp_train_responses_neg = np.arange(len(bodypart_desc_train_neg_all), dtype=np.float32)
    lock.release()
    
    PosSaveObject = SaveClass(bodypart_vote_train_pos_all_, bodypart_kp_train_responses_pos, bodypart_desc_train_samples_pos, training_bodypart)
    with open(options.train_data_pos, 'wb') as fin_save_pos:
        pickle.dump(PosSaveObject, fin_save_pos)

    NegSaveObject = SaveClass(bodypart_vote_train_neg_all_, bodypart_kp_train_responses_neg, bodypart_desc_train_samples_neg, training_bodypart)
    with open(options.train_data_neg, 'wb') as fin_save_neg:
        pickle.dump(NegSaveObject, fin_save_neg)



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
    parser.add_option("", "--nthread", dest="n_thread", type="int", default=1, help="maximum number of threads for multiprocessing")
    parser.add_option("", "--dir-images", dest="dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--dir-keypoints", dest="dir_keypoints", default="", help="directory to save keypoints")

    (options, args) = parser.parse_args()

    main(options, args)

