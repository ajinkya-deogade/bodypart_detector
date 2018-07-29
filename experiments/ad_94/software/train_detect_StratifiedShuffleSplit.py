#! /opt/local/bin/python

import json
import pickle
import random
import re
import time
from optparse import OptionParser
from pyflann import *
import os
import cv2
import numpy as np
import numpy.matlib
from sklearn.cross_validation import train_test_split
import csv

def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

class Error_Stats:
    def __init__(self):
        self.frame_file = None

class SaveClass:
    __slots__ = ['votes', 'keypoints', 'descriptors', 'bodypart', 'hessianThreshold', 'nOctaves', 'nOctaveLayers']
    def __init__(self, votes, keypoints, descriptors, bodypart, hessianThreshold, nOctaves, nOctaveLayers):
        self.votes = votes
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bodypart = bodypart
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers

surf = cv2.xfeatures2d.SURF_create(150, nOctaves=2, nOctaveLayers=3, extended=1)

def computeVoteMapOPENCV_oneNearestNeighbour(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma):
    bodypart_vote_map_op = {}
    for bid in range(0, len(detect_bodypart)):
        bodypart_vote_map_op[bid] = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float)

    bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
    for x in range(-vote_patch_size, vote_patch_size + 1):
        for y in range(-vote_patch_size, vote_patch_size + 1):
            bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

    kp_frame, desc_frame = surf.detectAndCompute(frame, None)
    if desc_frame is not None:
        for h, desc in enumerate(desc_frame):
            desc = np.array(desc, np.float32).reshape((1, 128))
            r_pos, d_pos = bodypart_knn_pos.nn_index(desc, 1, checks=8)
            r_pos = r_pos[0]
            a = np.pi * kp_frame[h].angle / 180.0
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

            for bi in range(0, len(bodypart_trained_data_pos.votes[r_pos])):
                vote_loc, vote_bodypart = bodypart_trained_data_pos.votes[r_pos][bi]
                vi = -1
                for vid in range(0, len(detect_bodypart)):
                    if ( detect_bodypart[vid] == vote_bodypart):
                        vi = vid
                        break
                if( vi == -1 ):
                    continue
                p = kp_frame[h].pt + np.dot(R, vote_loc)
                x, y = p
                if (not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or  y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size)):
                    y_start = int(float(y)) - int(float(vote_patch_size))
                    y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                    x_start = int(float(x)) - int(float(vote_patch_size))
                    x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                    bodypart_vote_map_op[vi][y_start:y_end, x_start:x_end] += bodypart_vote

    return bodypart_vote_map_op, len(kp_frame)

def computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma):
    bodypart_vote_map_op = {}
    for bid in range(0, len(detect_bodypart)):
        bodypart_vote_map_op[bid] = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float)

    bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
    for x in range(-vote_patch_size, vote_patch_size + 1):
        for y in range(-vote_patch_size, vote_patch_size + 1):
            bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

    kp_frame, desc_frame = surf.detectAndCompute(frame, None)
    if desc_frame is not None:
        for h, desc in enumerate(desc_frame):
            desc = np.array(desc, np.float32).reshape((1, 128))
            r_pos_all, d_pos_all = bodypart_knn_pos.nn_index(desc, 25, checks=8)
            r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, checks=8)

            for knn_id in range(0, np.shape(r_pos_all)[1]):
                r_pos = int(r_pos_all[:,knn_id])
                d_pos = d_pos_all[:,knn_id]
                relative_distance = d_pos - d_neg

                if (relative_distance <= desc_distance_threshold):
                    a = np.pi * kp_frame[h].angle / 180.0
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

                    for bi in range(0, len(bodypart_trained_data_pos.votes[r_pos])):
                        vote_loc, vote_bodypart = bodypart_trained_data_pos.votes[r_pos][bi]
                        vi = -1
                        for vid in range(0, len(detect_bodypart)):
                            if ( detect_bodypart[vid] == vote_bodypart):
                                vi = vid
                                break
                        if( vi == -1 ):
                            continue
                        p = kp_frame[h].pt + np.dot(R, vote_loc)
                        x, y = p
                        if (not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or  y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size)):
                            y_start = int(float(y)) - int(float(vote_patch_size))
                            y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                            x_start = int(float(x)) - int(float(vote_patch_size))
                            x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                            bodypart_vote_map_op[vi][y_start:y_end, x_start:x_end] += bodypart_vote

    return bodypart_vote_map_op, len(kp_frame)

def train(train_annotation_list, project_dir, train_bodypart, hessianThreshold, nOctaves, nOctaveLayers, pos_neg_equal, mh_neighborhood, crop_size):

    surf = cv2.xfeatures2d.SURF_create(int(hessianThreshold), nOctaves=int(nOctaves), nOctaveLayers=int(nOctaveLayers), extended=1)

    train_annotation = {}
    train_annotation["Annotations"] = []

    for train_annotation_file in train_annotation_list:
        train_annotation_file = os.path.join(project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
        with open(train_annotation_file) as fin_annotation:
            annotation_row = json.load(fin_annotation)
            train_annotation["Annotations"].extend(annotation_row["Annotations"])

    bodypart_kp_train_pos = []
    bodypart_desc_train_pos = []
    bodypart_vote_train_pos = []
    bodypart_kp_train_neg = []
    bodypart_desc_train_neg = []
    bodypart_vote_train_neg = []
    training_bodypart = train_bodypart
    keypoints_all = 0

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(project_dir, frame_file)
        frame = cv2.imread(frame_file)

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
                    if (np.sqrt(np.square(x - bodypart_coords[bodypart_id]["x"]) + np.square(y - bodypart_coords[bodypart_id]["y"])) <= mh_neighborhood):
                        flag_add_to_pos = True
                        bodypart_dp_pos = np.array([bodypart_coords[bodypart_id]["x"] - x, bodypart_coords[bodypart_id]["y"] - y]).T
                        bodypart_R_pos = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                        bodypart_dp_R_pos = np.dot(bodypart_R_pos, bodypart_dp_pos)
                        votes_tmp.append((bodypart_dp_R_pos, bodypart_coords[bodypart_id]["bodypart"]))

                if (flag_add_to_pos):
                    bodypart_kp_train_pos_frame.append(bodypart_kp[k])
                    bodypart_desc_train_pos_frame.append(bodypart_desc[k])
                    bodypart_vote_train_pos_frame.append(votes_tmp)

                flag_add_to_neg = False
                for bodypart_id in range(0, len(bodypart_coords)):
                    # if key point is greater than certain distance from the bodypart
                    if (np.sqrt(np.square(x - bodypart_coords[bodypart_id]["x"]) +
                                np.square(y - bodypart_coords[bodypart_id]["y"])) >= 1.5*mh_neighborhood):
                        flag_add_to_neg = True
                        break

                if (flag_add_to_neg):
                    bodypart_kp_train_neg_frame.append(bodypart_kp[k])
                    bodypart_desc_train_neg_frame.append(bodypart_desc[k])

            if pos_neg_equal:
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

        os.system('cls')
        print "Training Body Part: ", train_bodypart
        print "Percentage Complete: %.2f" %(float(i)/float(len(train_annotation["Annotations"]))*100)

    print "Number of Positive Training Samples: ", len(bodypart_kp_train_pos)
    print "Number of Negative Training Samples: ", len(bodypart_kp_train_neg)
    print "Number of Actual Keypoints Used: ", keypoints_all

    bodypart_desc_train_samples_pos = np.array(bodypart_desc_train_pos)
    bodypart_kp_train_responses_pos = np.arange(len(bodypart_kp_train_pos), dtype=np.float32)
    bodypart_desc_train_samples_neg = np.array(bodypart_desc_train_neg)
    bodypart_kp_train_responses_neg = np.arange(len(bodypart_kp_train_neg), dtype=np.float32)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    timeStampFolder = '../expts/' + timestr
    if not os.path.exists(timeStampFolder):
        os.makedirs(timeStampFolder)

    timeStampFolder_Python = timeStampFolder + '/python/'
    if not os.path.exists(timeStampFolder_Python):
        os.makedirs(timeStampFolder_Python)
    train_data_pos = os.path.join(timeStampFolder_Python, 'positive.p')
    train_data_neg = os.path.join(timeStampFolder_Python, 'negative.p')

    print 'bodypart_vote_train_pos', np.shape(bodypart_vote_train_pos)
    print 'bodypart_desc_train_samples_pos', np.shape(bodypart_desc_train_samples_pos)
    print 'bodypart_kp_train_pos', np.shape(bodypart_kp_train_pos)
    allRows = []

    i = -1
    for votes in bodypart_vote_train_pos:
        i += 1
        row = []
        row.extend(bodypart_desc_train_pos[i])
        row_tmp = np.zeros((1, len(train_bodypart) * 3))
        # print 'Vote : ', votes
        for vote in votes:
            if vote[1] in train_bodypart:
                pos = train_bodypart.index(vote[1])
                pos2 = 3 * pos
                row_tmp[0, pos2] = vote[0][0]
                row_tmp[0, pos2 + 1] = vote[0][1]
                row_tmp[0, pos2 + 2] = 1
        row.extend(row_tmp[0])
        allRows.append(row)

    print 'Total Number of Rows : ', i+1
    print 'Length of Rows Before Filter : ', np.shape(allRows)
    allRows_2 = np.vstack({tuple(row) for row in allRows})
    print 'Length of Rows After Filter : ', np.shape(allRows_2)

    bodypart_desc_train_samples_pos = []
    bodypart_vote_train_pos = []
    for row in allRows_2:
        bodypart_desc_train_samples_pos.append(row[:128])
        votes_tmp = []
        tempRLdata = row[128:]
        k = -1
        for flag in range(2, len(tempRLdata), 3):
            k += 1
            if tempRLdata[flag] == 1:
                rel = np.array([tempRLdata[flag-2], tempRLdata[flag-1]], dtype=np.float32)
                votes_tmp.append((rel, train_bodypart[k]))

        bodypart_vote_train_pos.append(votes_tmp)

    bodypart_kp_train_responses_pos = np.arange(np.shape(bodypart_vote_train_pos)[0], dtype=np.float32)
    bodypart_vote_train_pos = np.squeeze(bodypart_vote_train_pos)
    print 'bodypart_kp_train_responses_pos : ', np.shape(bodypart_kp_train_responses_pos)
    print 'bodypart_vote_train_pos : ', np.shape(bodypart_vote_train_pos)
    print 'bodypart_desc_train_samples_pos : ', np.shape(bodypart_desc_train_samples_pos)

    PosSaveObject = SaveClass(bodypart_vote_train_pos, bodypart_kp_train_responses_pos, bodypart_desc_train_samples_pos, training_bodypart, int(hessianThreshold), int(nOctaves), int(nOctaveLayers))
    with open(train_data_pos, 'wb') as fin_save_pos:
        pickle.dump(PosSaveObject, fin_save_pos)

    NegSaveObject = SaveClass(bodypart_vote_train_neg, bodypart_kp_train_responses_neg, bodypart_desc_train_samples_neg, training_bodypart, int(hessianThreshold), int(nOctaves), int(nOctaveLayers))
    with open(train_data_neg, 'wb') as fin_save_neg:
        pickle.dump(NegSaveObject, fin_save_neg)

    print "Finished Training ........"

    trainList = os.path.join(timeStampFolder_Python, timestr + '_trainList.lst')
    print 'trainList', trainList
    listWriter = open(trainList, 'w')
    for trainFile in train_annotation_list:
        listWriter.write(trainFile)
    listWriter.close()

    trainFileName = os.path.join(timeStampFolder_Python, timestr + '_trainingParameters.json')
    print 'trainFileName', trainFileName
    results = open(trainFileName, 'w')
    trainData = {}
    trainData["TrainAnnotationListFile"] = trainList
    trainData["HessianThreshold"] =  hessianThreshold
    trainData["nOctaves"] =  nOctaves
    trainData["nOctaveLayers"] = nOctaveLayers
    trainData["EqualizePosNegKP"] = pos_neg_equal
    trainData["BodyPartNeightbourhoodDistance"] = mh_neighborhood
    trainData["CropSize"] = crop_size
    trainData["NumberPositiveKeypoint"] = len(bodypart_kp_train_pos)
    trainData["NumberNegativeKeypoint"] = len(bodypart_kp_train_neg)
    trainData["NumberKeypointPresent"] = keypoints_all
    json.dump(trainData, results)
    results.close()

    return train_data_pos, train_data_neg, timeStampFolder_Python

def detect(detect_bodypart, project_dir, train_data_p, train_data_n, test_annotation_list, vote_sigma, vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, timeStampFolder, kNN_algo):

    print "Performing detections......."
    test_annotations = []

    for test_annotation_file in test_annotation_list:
        test_annotation_file = os.path.join(project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
        with open(test_annotation_file) as fin_annotation:
            test_annotation = json.load(fin_annotation)
            test_annotations.extend(test_annotation["Annotations"])
    print "len(test_annotations):" , len(test_annotations)

    frame_index = -1
    misdectectedFrames = []
    error_stats_all = {}
    n_instance_gt = {}
    n_instance_gt["MouthHook"] = 0
    incorrectDetections = {}
    for s in detect_bodypart:
        error_stats_all[s] = []
        n_instance_gt[s] = 0
        incorrectDetections[s] = []

    bodypart_gt = {}
    verbosity = 0
    display_level = 0

    bodypart_trained_data_pos = pickle.load(open(train_data_p, 'rb'))
    bodypart_trained_data_neg = pickle.load(open(train_data_n, 'rb'))

    bodypart_knn_pos = FLANN()
    bodypart_knn_pos.build_index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32))
    bodypart_knn_neg = FLANN()
    bodypart_knn_neg.build_index(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float32))

    headers = ['FrameNumber', 'Time', 'Stage_x', 'Stage_y', 'Centroid_x', 'Centroid_y', 'Midpoint_x', 'Midpoint_y', 'Head_x', 'Head_y', 'Tail_x', 'Tail_y',
               'MouthHook_x', 'MouthHook_y', 'LeftMHhook_x', 'LeftMHhook_y', 'RightMHhook_x', 'RightMHhook_y', 'LeftDorsalOrgan_x', 'LeftDorsalOrgan_y',
               'RightDorsalOrgan_x', 'RightDorsalOrgan_y', 'CenterBolwigOrgan_x', 'CenterBolwigOrgan_y', 'LeftBolwigOrgan_x', 'LeftBolwigOrgan_y',
               'RightBolwigOrgan_x', 'RightBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y',
               'RightBolwigOrgan_x', 'RightBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y',
               'RightBolwigOrgan_x', 'RightBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y',
               'MouthHook_conf', 'LeftMHhook_conf', 'RightMHhook_conf', 'LeftDorsalOrgan_conf', 'RightDorsalOrgan_conf', 'CenterBolwigOrgan_conf',
               'LeftBolwigOrgan_conf', 'RightBolwigOrgan_conf', 'RightBolwigOrgan_conf', 'RightBolwigOrgan_conf', 'RightBolwigOrgan_conf', 'RightBolwigOrgan_conf',
               'RightBolwigOrgan_conf', 'RightBolwigOrgan_conf', 'RightBolwigOrgan_conf', 'RightBolwigOrgan_conf', 'NumberKeypoints']

    annotationMatrix = []
    detectionMatrix = []


    for j in range(0, len(test_annotations)):
        frame_index += 1
        os.system('cls')
        print "Percentage Complete: %.2f" %(float(frame_index)/float(len(test_annotations))*100)

        annotation = test_annotations[j]

        frame_file_original = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file_original)
        frame_file = os.path.join(project_dir, frame_file)
        frame = cv2.imread(frame_file)

        if (display_level >= 2):
            display_voters = frame.copy()

        flag_skip = True
        bodypart_coords_gt = {}
        for k in range(0, len(annotation["FrameValueCoordinates"])):
            bi = annotation["FrameValueCoordinates"][k]["Name"]
            if ((bi == "MouthHook" or any(bi == s for s in detect_bodypart)) and annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"] != -1):
                flag_skip = False
                bodypart_coords_gt[bi] = {}
                bodypart_coords_gt[bi]["x"] = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                bodypart_coords_gt[bi]["y"] = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])
                n_instance_gt[bi] += 1

        if ( flag_skip ):
            continue

        bodypart_gt[frame_index] = {}
        bodypart_gt[frame_index]["bodypart_coords_gt"] = bodypart_coords_gt
        bodypart_gt[frame_index]["frame_file"] = frame_file

        if (verbosity >= 2):
            print "bodypart_coords_gt:" , bodypart_coords_gt

        annotation_row = {}
        detection_row = {}
        for m in headers:
            annotation_row[m] = -1
            detection_row[m] = -1
        for j in range(0, len(annotation["FrameValueCoordinates"])):
            annotation_row['FrameNumber'] = int(annotation["FrameIndexVideo"])
            bpName = annotation["FrameValueCoordinates"][j]["Name"]
            bp_x = annotation["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
            bp_y = annotation["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
            if (bpName == "MouthHook"):
                annotation_row['MouthHook_x'] = bp_x
                annotation_row['MouthHook_y'] = bp_y
                annotation_row['MouthHook_conf'] = 100
            if (bpName == "LeftMHhook"):
                annotation_row['LeftMHhook_x'] = bp_x
                annotation_row['LeftMHhook_y'] = bp_y
                annotation_row['LeftMHhook_conf'] = 100
            if (bpName == "RightMHhook"):
                annotation_row['RightMHhook_x'] = bp_x
                annotation_row['RightMHhook_y'] = bp_y
                annotation_row['RightMHhook_conf'] = 100
            if (bpName == "LeftDorsalOrgan"):
                annotation_row['LeftDorsalOrgan_x'] = bp_x
                annotation_row['LeftDorsalOrgan_y'] = bp_y
                annotation_row['LeftDorsalOrgan_conf'] = 100
            if (bpName == "RightDorsalOrgan"):
                annotation_row['RightDorsalOrgan_x'] = bp_x
                annotation_row['RightDorsalOrgan_y'] = bp_y
                annotation_row['RightDorsalOrgan_conf'] = 100
        annotationMatrix.append(annotation_row)

        if "MouthHook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
            crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-int(crop_size/2))
            crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-int(crop_size/2))
            frame = frame[crop_y:crop_y+crop_size,crop_x:crop_x+crop_size,0]

            image_info = np.shape(frame)
            if (verbosity >= 2):
                print image_info

            bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
            for x in range(-vote_patch_size, vote_patch_size + 1):
                for y in range(-vote_patch_size, vote_patch_size + 1):
                    bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

            if (verbosity >= 2):
                print "Bodypart Vote: ", np.shape(bodypart_vote)

            ack_message={}
            bodypart_coords_est = {}

            image_header = {}
            image_header["rows"] = image_info[0]
            image_header["cols"] = image_info[1]
            image_header["crop_x"] = crop_x
            image_header["crop_y"] = crop_y

            if (verbosity >= 2):
                print "image num rows:", image_header["rows"]
                print "image num cols:", image_header["cols"]
                print "image header origin-x-coord:", image_header["crop_x"]
                print "image header origin-y-coord:", image_header["crop_y"]

            bodypart_vote_map = []
            for bid in range(0, len(detect_bodypart)):
                bodypart_vote_map.append(np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float))

            if (display_level >= 2):
                display_voters = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if kNN_algo == 1:
                bodypart_vote_map, numberKP = computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma)
            elif kNN_algo == 0:
                bodypart_vote_map, numberKP = computeVoteMapOPENCV_oneNearestNeighbour(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma)
        else:
            continue

        if (display_level >= 2):
            display_voters = cv2.resize(display_voters, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("voters", display_voters)

        for bi in range(0, len(detect_bodypart)):
            vote_max = np.amax(bodypart_vote_map[bi][:,:])
            if (vote_max > vote_threshold and ((detect_bodypart[bi] not in bodypart_coords_est) or vote_max > bodypart_coords_est[detect_bodypart[bi]]["conf"])):
                vote_max_loc = np.array(np.where(bodypart_vote_map[bi][:,:] == vote_max))
                vote_max_loc = vote_max_loc[:,0]
                bodypart_coords_est[detect_bodypart[bi]] = {"conf" : vote_max,
                                                                    "x" : int(vote_max_loc[1]) + int(image_header["crop_x"]),
                                                                    "y" : int(vote_max_loc[0]) + int(image_header["crop_y"])}

        if (display_level >= 2):
            display_vote_map = np.array(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).copy(), np.float)
            display_vote_map /= 255.0
            bodypart_vote_map /= np.amax(bodypart_vote_map)
            display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
            if ("x" in bodypart_coords_est):
                cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
            display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("voters", display_vote_map)

        ack_message["detections"] = []
        for bi in detect_bodypart:
            if ( bi in bodypart_coords_est ):
                ack_message["detections"].append( { "frame_index" : frame_index,
                                                    "test_bodypart" : bi,
                                                    "coord_x" : bodypart_coords_est[bi]["x"],
                                                    "coord_y" : bodypart_coords_est[bi]["y"],
                                                    "conf" : bodypart_coords_est[bi]["conf"]} )
        ack_message = json.dumps(ack_message, separators=(',',':'))
        received_json = json.loads(ack_message)

        frame_error_dist = np.zeros((1, 8))
        if ( "detections" in received_json ):
            for di in range(0, len(received_json["detections"])):
                tbp = received_json["detections"][di]["test_bodypart"]
                fi = received_json["detections"][di]["frame_index"]
                bp_x = received_json["detections"][di]["coord_x"]
                bp_y = received_json["detections"][di]["coord_y"]
                conf = received_json["detections"][di]["conf"]
                detection_row['FrameNumber'] = fi
                if (tbp == "MouthHook"):
                    detection_row['MouthHook_x'] = bp_x
                    detection_row['MouthHook_y'] = bp_y
                    detection_row['MouthHook_conf'] = conf
                if (tbp == "LeftMHhook"):
                    detection_row['LeftMHhook_x'] = bp_x
                    detection_row['LeftMHhook_y'] = bp_y
                    detection_row['LeftMHhook_conf'] = conf
                if (tbp == "RightMHhook"):
                    detection_row['RightMHhook_x'] = bp_x
                    detection_row['RightMHhook_y'] = bp_y
                    detection_row['RightMHhook_conf'] = conf
                if (tbp == "LeftDorsalOrgan"):
                    detection_row['LeftDorsalOrgan_x'] = bp_x
                    detection_row['LeftDorsalOrgan_y'] = bp_y
                    detection_row['LeftDorsalOrgan_conf'] = conf
                if (tbp == "RightDorsalOrgan"):
                    detection_row['RightDorsalOrgan_x'] = bp_x
                    detection_row['RightDorsalOrgan_y'] = bp_y
                    detection_row['RightDorsalOrgan_conf'] = conf
                if (tbp == "CenterBolwigOrgan"):
                    detection_row['CenterBolwigOrgan_x'] = bp_x
                    detection_row['CenterBolwigOrgan_y'] = bp_y
                    detection_row['CenterBolwigOrgan_conf'] = conf
                if (tbp == "LeftBolwigOrgan"):
                    detection_row['LeftBolwigOrgan_x'] = bp_x
                    detection_row['LeftBolwigOrgan_y'] = bp_y
                    detection_row['LeftBolwigOrgan_conf'] = conf
                if (tbp == "RightBolwigOrgan"):
                    detection_row['RightBolwigOrgan_x'] = bp_x
                    detection_row['RightBolwigOrgan_y'] = bp_y
                    detection_row['RightBolwigOrgan_conf'] = conf

                if (tbp in bodypart_gt[fi]["bodypart_coords_gt"]):
                    error_stats = Error_Stats()
                    error_stats.frame_file =  bodypart_gt[fi]["frame_file"]
                    error_stats.error_distance = np.sqrt(np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"] - received_json["detections"][di]["coord_x"]) + np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"] - received_json["detections"][di]["coord_y"]))
                    error_stats.conf = received_json["detections"][di]["conf"]
                    frame_error_dist[0, train_bodypart.index(tbp)] = error_stats.error_distance
                    error_stats_all[tbp].append(error_stats)
                    if error_stats.error_distance > outlier_error_dist:
                        localAnnotation = {}
                        localAnnotation['FrameFile'] = frame_file_original
                        localAnnotation['annotated_x'] = bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"]
                        localAnnotation['annotated_y'] = bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"]
                        localAnnotation['detected_x'] = received_json["detections"][di]["coord_x"]
                        localAnnotation['detected_y'] = received_json["detections"][di]["coord_y"]
                        localAnnotation['error'] = error_stats.error_distance
                        incorrectDetections[tbp].append(localAnnotation)

            if np.any(frame_error_dist > outlier_error_dist):
                misdectectedFrames.append(frame_file_original)

            detectionMatrix.append(detection_row)

    print os.sys.argv
    timestr = time.strftime("%Y%m%d_%H%M%S")

    testList = os.path.join(timeStampFolder, timestr + '_testList.lst')
    listWriter = open(testList, 'w')
    for testFile in test_annotation_list:
        listWriter.write(testFile)
    listWriter.close()

    testData = {}
    testData["DetectionParameters"] = {}
    testData["DetectionParameters"]["PositiveTrainFile"] = train_data_p
    testData["DetectionParameters"]["NegativeTrainFile"] = train_data_n
    testData["DetectionParameters"]["TestAnnotationListFile"] = testList
    testData["DetectionParameters"]["VoteSigma"] = vote_sigma
    testData["DetectionParameters"]["VotePatchSize"] = vote_patch_size
    testData["DetectionParameters"]["VoteDescriptorErrorDistance"] =  desc_distance_threshold
    testData["DetectionParameters"]["VoteThreshold" ] = vote_threshold
    testData["DetectionParameters"]["OutlierErrorDistance"] =  outlier_error_dist

    testData["DetectionResults"] = {}
    for bid in error_stats_all:
        testData["DetectionResults"][bid] = {}
        error_distance_inliers = []
        inlier_confs = []
        outlier_confs = []
        for es in error_stats_all[bid]:
            if (es.error_distance <= outlier_error_dist):
                error_distance_inliers.append(es.error_distance)
                inlier_confs.append(es.conf)
            else:
                outlier_confs.append(es.conf)

        n_outlier = n_instance_gt[bid] - len(error_distance_inliers)
        testData["DetectionResults"][bid]["NumberInlier"] = len(error_distance_inliers)
        testData["DetectionResults"][bid]["InlierErrorDistance"] = error_distance_inliers
        testData["DetectionResults"][bid]["InlierConfidence"] = inlier_confs

        testData["DetectionResults"][bid]["NumberOutlier"] = n_outlier
        testData["DetectionResults"][bid]["OutlierConfidence"] = outlier_confs

        testData["DetectionResults"][bid]["GroundTruthInstance"] = n_instance_gt[bid]
        testData["DetectionResults"][bid]["NumberDetection"] = len(error_stats_all[bid])
        testData["DetectionResults"][bid]["ProportionOutlier"] = float(n_outlier) / max(1, float(n_instance_gt[bid]))

        print "Body part:", bid
        print "Number of inliers: ", len(error_distance_inliers)
        print "Ground truth number of instances:", n_instance_gt[bid]
        print "Total number of detections:", len(error_stats_all[bid])
        print "Proportion of inliers (within %d) = %d / %d = %g" % (outlier_error_dist, len(error_distance_inliers), n_instance_gt[bid], float(len(error_distance_inliers)) / max(1, float(n_instance_gt[bid])) )

        if (len(error_distance_inliers) > 0):
            testData["DetectionResults"][bid]["MedianInlierErrorDist"] = np.median(error_distance_inliers)
            testData["DetectionResults"][bid]["MeanInlierErrorDist"] = np.mean(error_distance_inliers)
            testData["DetectionResults"][bid]["MinInlierConfidence"] = np.min(inlier_confs)
            testData["DetectionResults"][bid]["MeanInlierConfidence"] = np.mean(inlier_confs)

            print "Median inlier error dist =", np.median(error_distance_inliers)
            print "Mean inlier error dist =", np.mean(error_distance_inliers)
            print "Min inlier confidence =", np.min(inlier_confs)
            print "Mean inlier confidence =", np.mean(inlier_confs)
        else:
            testData["DetectionResults"][bid]["MedianInlierErrorDist"] = []
            testData["DetectionResults"][bid]["MeanInlierErrorDist"] = []
            testData["DetectionResults"][bid]["MinInlierConfidence"] = []
            testData["DetectionResults"][bid]["MeanInlierConfidence"] = []

        if (len(outlier_confs) > 0):
            testData["DetectionResults"][bid]["MaxOutlierConfidence"] = np.max(outlier_confs)
            testData["DetectionResults"][bid]["MeanOutlierConfidence"]  = np.mean(outlier_confs)

            print "Max outlier confidence = ", np.max(outlier_confs)
            print "Mean outlier confidence = ", np.mean(outlier_confs)
        else:
            testData["DetectionResults"][bid]["MaxOutlierConfidence"] = []
            testData["DetectionResults"][bid]["MeanOutlierConfidence"]  = []

    resultFileName = os.path.join(timeStampFolder, timestr + '_results.log')
    res = open(resultFileName, 'w')
    json.dump(testData, res)
    res.close()

    incorrectDetectionFileName = os.path.join(timeStampFolder, timestr + '_incorrectDetections.json')
    resHab = open(incorrectDetectionFileName, 'w')
    json.dump(incorrectDetections, resHab)
    resHab.close()

    annotationFilename = os.path.join(timeStampFolder, timestr + '_annotations.csv')
    annotationWriter = csv.DictWriter(open(annotationFilename, 'w'), delimiter=',', lineterminator='\n', fieldnames=headers)
    annotationWriter.writerows(annotationMatrix)

    detectionsFilename = os.path.join(timeStampFolder, timestr + '_detections.csv')
    detectionWriter = csv.DictWriter(open(detectionsFilename, 'w'), delimiter=',', lineterminator='\n', fieldnames=headers)
    detectionWriter.writeheader()
    detectionWriter.writerows(detectionMatrix)

    testList = os.path.join(timeStampFolder, timestr + '_badFrames.lst')
    writer = csv.writer(open(testList, 'wb'))
    for missedFile in misdectectedFrames:
        writer.writerow([missedFile])
    listWriter.close()

if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list-all", dest="train_annotation_list_all", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--test-annotation-list-all", dest="test_annotation_list_all", default="", help="list of frame level training annotation JSON files")
    parser.add_option("", "--mis-detected-list-all", dest="mis_detected_list_all", default="", help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--mh-neighborhood", dest="mh_neighborhood", type="int", default=100, help="distance from mouth hook for a keyppoint to be considered relevant for training")
    parser.add_option("", "--positive-training-datafile", dest="train_data_pos", help="File to save the information about the positive training data")
    parser.add_option("", "--negative-training-datafile", dest="train_data_neg", help="File to save the information about the negative training data")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results.write visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--training-bodypart", dest="train_bodypart", default="MouthHook", action="callback", type="string", callback=string_split, help="Input the bodypart to be trained")
    parser.add_option("", "--nOctaves", dest="nOctaves", default=2, type="int", help="Input the number of octaves used in surf object")
    parser.add_option("", "--nOctaveLayers", dest="nOctaveLayers", default=3, type="int", help="Input the number of octave layers used in surf object")
    parser.add_option("", "--hessian-threshold", dest="hessianThreshold", default=250, type="int", help="Input the bodypart to be trained")
    parser.add_option("", "--pos-neg-equal", dest="pos_neg_equal", default=1, type="int", help="Input the bodypart to be trained")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.0, help="threshold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15, help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=5.0, help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--vote-threshold", dest="vote_threshold", type="float", default=0.0, help="threshold on the net vote for a location for it to be a viable detection")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=7,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--crop-size", dest="crop_size", type="int", default=256,help="Crops surrounding Mouthhook")
    parser.add_option("", "--kNN-algo", dest="kNN_algo", type="int", default=1, help="Crops surrounding Mouthhook")

    (options, args) = parser.parse_args()

    train_annotation_list_all = options.train_annotation_list_all
    test_annotation_list_all = options.test_annotation_list_all
    project_dir = options.project_dir
    train_bodypart = options.train_bodypart
    hessianThreshold = options.hessianThreshold
    nOctaves = options.nOctaves
    nOctaveLayers = options.nOctaveLayers
    pos_neg_equal = options.pos_neg_equal
    mh_neighborhood = options.mh_neighborhood
    vote_sigma = options.vote_sigma
    vote_patch_size = options.vote_patch_size
    desc_distance_threshold = options.desc_distance_threshold
    vote_threshold = options.vote_threshold
    outlier_error_dist = options.outlier_error_dist
    kNN_algo = options.kNN_algo
    mis_detected_list_all = options.mis_detected_list_all

    crop_size = options.crop_size

    train_list_pos_all = []
    train_list_neg_all = []

    X = []
    with open(train_annotation_list_all) as all_list:
        for train_annotation_list in all_list:
            X.append(train_annotation_list)

    train_annotation_list = X
    # print train_annotation_list

    train_flag = 0
    if train_flag > 0:
        train_data_positive, train_data_negative, timeStampFolder = train(train_annotation_list, project_dir, train_bodypart, hessianThreshold, nOctaves, nOctaveLayers, pos_neg_equal, mh_neighborhood, crop_size)
    else:
        train_data_positive = '../expts/all_7_except_Rawdata_20170317_194352/python/positive.p'
        train_data_negative = '../expts/all_7_except_Rawdata_20170317_194352/python/negative.p'
        timeStampFolder = '../expts/all_7_except_Rawdata_20170317_194352/python/'

        # train_data_positive = '../expts/20170303_230839_only914_RemovedDuplicates/python/positive.p'
        # train_data_negative = '../expts/20170303_230839_only914_RemovedDuplicates/python/negative.p'
        # timeStampFolder = '../expts/20170303_230839_only914_RemovedDuplicates/python/'

    detect_bodypart = train_bodypart

    # testList = '../config/trainAndTestOnSame'
    # testList = '../config/trainUsingSameDay_20170224Data'
    # testList = '../config/temp'

    Y = []
    with open(test_annotation_list_all) as all_list:
        for test_annotation_list in all_list:
            Y.append(test_annotation_list)

    test_annotation_list = Y

    detect(detect_bodypart, project_dir, train_data_positive, train_data_negative, test_annotation_list, vote_sigma,
           vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, timeStampFolder, kNN_algo)