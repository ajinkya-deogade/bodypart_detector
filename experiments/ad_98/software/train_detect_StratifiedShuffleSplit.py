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
import csv
import itertools
import pandas as pd

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

# surf = cv2.xfeatures2d.SURF_create(150, nOctaves=2, nOctaveLayers=3, extended=1)

def computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma, kp_frame, desc_frame, detect_bodypart):
    bodypart_vote_map_op = {}
    for bid in range(0, len(detect_bodypart)):
        bodypart_vote_map_op[bid] = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float)

    bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
    for x in range(-vote_patch_size, vote_patch_size + 1):
        for y in range(-vote_patch_size, vote_patch_size + 1):
            bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

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
                    a = kp_frame[h].angle
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    # allBP = bodypart_trained_data_pos.bodypart

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

def readFPGA_KP_DescfromFile(kp_file, desc_file):
    ## Read Keypoints
    keypoints = []
    with open(kp_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
            row = row[0]
            keypoints_temp = cv2.KeyPoint(x=float(row[0]), y=float(row[1]),
                                          _size=float((float(row[2]) / float(1.2)) * float(9)),
                                          _angle=float(2*np.pi - float(row[6])), _response=float(row[3]),
                                          _octave=int(row[4]), _class_id=int(row[5]))
            keypoints.append(keypoints_temp)

    ## Read Descriptors
    descriptors = []
    with open(desc_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
            row = row[0]
            descriptors.append(row)

    return keypoints, descriptors

def readJSON_to_CSV(fin_annotation_list, timeStampFolder):
    headers = ['FrameNumber','MouthHook_x','MouthHook_y','LeftMHhook_x','LeftMHhook_y','RightMHhook_x','RightMHhook_y','LeftDorsalOrgan_x','LeftDorsalOrgan_y','RightDorsalOrgan_x','RightDorsalOrgan_y',
               'CenterBolwigOrgan_x', 'CenterBolwigOrgan_y', 'LeftBolwigOrgan_x', 'LeftBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y']
    for train_annotation_file in fin_annotation_list:
        with open(train_annotation_file.strip()) as fin_annotation:
            train_annotation = []
            save_folder = timeStampFolder
            save_name = os.path.join(save_folder, os.path.splitext(os.path.basename(train_annotation_file))[0]) + ".csv"
            writer = csv.DictWriter(open(save_name, 'w'), delimiter=',', lineterminator='\n', fieldnames=headers)

            tmp_train_annotation = json.load(fin_annotation)
            for i in range(0, len(tmp_train_annotation["Annotations"])):
                temp_row = {}
                for m in headers:
                    temp_row[m] = None
                for j in range(0, len(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"])):
                    temp_row['FrameNumber'] = int(tmp_train_annotation["Annotations"][i]["FrameIndexVideo"])
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook"):
                        temp_row['MouthHook_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['MouthHook_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "LeftMHhook"):
                        temp_row['LeftMHhook_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['LeftMHhook_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightMHhook"):
                        temp_row['RightMHhook_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['RightMHhook_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j][
                        "Name"] == "LeftDorsalOrgan"):
                        temp_row['LeftDorsalOrgan_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['LeftDorsalOrgan_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j][
                        "Name"] == "RightDorsalOrgan"):
                        temp_row['RightDorsalOrgan_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['RightDorsalOrgan_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j][
                        "Name"] == "CenterBolwigOrgan"):
                        temp_row['CenterBolwigOrgan_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['CenterBolwigOrgan_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j][
                        "Name"] == "LeftBolwigOrgan"):
                        temp_row['LeftBolwigOrgan_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['LeftBolwigOrgan_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                    if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j][
                        "Name"] == "RightBolwigOrgan"):
                        temp_row['RightBolwigOrgan_x'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                        temp_row['RightBolwigOrgan_y'] = \
                        tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                train_annotation.append(temp_row)
            # print "Train Annotation: ", train_annotation
            writer.writerows(train_annotation)

def train(train_annotation_list, project_dir, train_bodypart, hessianThreshold, nOctaves, nOctaveLayers, pos_neg_equal, mh_neighborhood, crop_size, dir_keypoints, dir_descriptors, timeStampFolder):

    train_annotation = {}
    train_annotation["Annotations"] = []

    for train_annotation_file in train_annotation_list:
        train_annotation_file = os.path.join(project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
        with open(train_annotation_file) as fin_annotation:
            tmp_train_annotation = json.load(fin_annotation)
            train_annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    bodypart_kp_train_pos = []
    bodypart_desc_train_pos = []
    bodypart_vote_train_pos = []
    bodypart_kp_train_neg = []
    bodypart_desc_train_neg = []
    bodypart_vote_train_neg = []
    training_bodypart = train_bodypart
    keypoints_all = 0
    fpgaTable = []
    bpKPcount = {}

    for s in train_bodypart:
        bpKPcount[s] = 0

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
            # frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, 0]

            try:
                ## Read Keypoints
                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)
                keypoints_folder = os.path.join(dir_keypoints, parent_dir)
                if not os.path.exists(keypoints_folder):
                    print "Folder does not exist !!!"

                keypoints_file = os.path.join(keypoints_folder, os.path.splitext(os.path.basename(train_annotation["Annotations"][i]["FrameFile"]))[0]) + ".csv"

                ## Read Descriptors
                current_dir = os.path.abspath(os.path.dirname(frame_file))
                parent_dir = os.path.basename(current_dir)
                descriptors_folder = os.path.join(dir_descriptors, parent_dir)
                if not os.path.exists(descriptors_folder):
                    print "Folder does not exist !!!"

                descriptors_file = os.path.join(descriptors_folder, os.path.splitext(os.path.basename(train_annotation["Annotations"][i]["FrameFile"]))[0]) + ".csv"
                bodypart_kp, bodypart_desc = readFPGA_KP_DescfromFile(keypoints_file, descriptors_file)

                for k in range(0, len(bodypart_kp)):
                    keypoints_all = keypoints_all + 1
                    x, y = bodypart_kp[k].pt
                    x = x + crop_x
                    y = y + crop_y
                    a = bodypart_kp[k].angle

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
                            bpKPcount[bodypart_coords[bodypart_id]["bodypart"]] += 1

                    if ( flag_add_to_pos ):
                        bodypart_kp_train_pos_frame.append(bodypart_kp[k])
                        bodypart_desc_train_pos_frame.append(bodypart_desc[k])
                        bodypart_vote_train_pos_frame.append(votes_tmp)

                        fpgaTableRow_tmp = []
                        fpgaTableRow_tmp.append(sum(bodypart_desc[k]))
                        fpgaTableRow_tmp.extend(bodypart_desc[k])
                        bp_present = np.zeros((1, len(train_bodypart)))
                        fpgaVotes_tmp = np.zeros((1, len(train_bodypart)*3))
                        for vote in votes_tmp:
                            if vote[1] in train_bodypart:
                                pos = train_bodypart.index(vote[1])
                                pos2 = 3*pos
                                bp_present[0, pos] = 1
                                fpgaVotes_tmp[0, pos2] = round(vote[0][0], 4)
                                fpgaVotes_tmp[0, pos2+1] = round(vote[0][1], 4)
                                fpgaVotes_tmp[0, pos2+2] = 1

                        if np.any(np.squeeze(bp_present)):
                            fpgaVotes_tmp = np.squeeze(fpgaVotes_tmp)
                            fpgaTableRow_tmp.extend(fpgaVotes_tmp)

                        fpgaTable.append(fpgaTableRow_tmp)

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
            except:
                print 'Error Reading File '
                continue

            os.system('clear')
            print "Training Body Part: ", train_bodypart
            print "Percentage Complete: %.2f" %(float(i)/float(len(train_annotation["Annotations"]))*100)


    for s in train_bodypart:
        print '%s KP count : %d'%(s, bpKPcount[s])

    print "Number of Positive Training Samples: ", len(bodypart_kp_train_pos)
    print "Number of Negative Training Samples: ", len(bodypart_kp_train_neg)
    print "Number of Actual Keypoints Used: ", keypoints_all

    # bodypart_desc_train_samples_pos = np.array(bodypart_desc_train_pos)
    # bodypart_kp_train_responses_pos = np.arange(len(bodypart_kp_train_pos), dtype=np.float32)
    # bodypart_desc_train_samples_neg = np.array(bodypart_desc_train_neg)
    # bodypart_kp_train_responses_neg = np.arange(len(bodypart_kp_train_neg), dtype=np.float32)

    timeStampFolder_fpga = timeStampFolder
    if not os.path.exists(timeStampFolder_fpga):
        os.makedirs(timeStampFolder_fpga)
    train_data_pos = os.path.join(timeStampFolder_fpga, 'positive.p')
    train_data_neg = os.path.join(timeStampFolder_fpga, 'negative.p')

    # PosSaveObject = SaveClass(bodypart_vote_train_pos, bodypart_kp_train_responses_pos, bodypart_desc_train_samples_pos, training_bodypart, int(hessianThreshold), int(nOctaves), int(nOctaveLayers))
    # with open(train_data_pos, 'wb') as fin_save_pos:
    #     pickle.dump(PosSaveObject, fin_save_pos)
    #
    # NegSaveObject = SaveClass(bodypart_vote_train_neg, bodypart_kp_train_responses_neg, bodypart_desc_train_samples_neg, training_bodypart, int(hessianThreshold), int(nOctaves), int(nOctaveLayers))
    # with open(train_data_neg, 'wb') as fin_save_neg:
    #     pickle.dump(NegSaveObject, fin_save_neg)

    print "Finished Training ........"

    trainList = os.path.join(timeStampFolder_fpga, timestr + '_trainList.lst')
    # print 'trainList', trainList
    # listWriter = open(trainList, 'w')
    # for trainFile in train_annotation_list:
    #     listWriter.write(trainFile)
    # listWriter.close()

    trainFileName = os.path.join(timeStampFolder_fpga, timestr + '_trainingParameters.json')
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

    trainFileName = os.path.join(timeStampFolder_fpga, timestr + '_FPGA_trainData.txt')
    fpgaTable = np.asarray(fpgaTable)
    fpgaTable = np.squeeze(fpgaTable)
    fpgaTable = sorted(fpgaTable, key=lambda x: x[0])
    fpgaTable = np.vstack({tuple(row) for row in fpgaTable})
    fpgaTable_pd = pd.DataFrame(fpgaTable)
    fpgaTable_pd.sort_values(by=[0], inplace=True)
    fpgaTable_pd.to_csv(trainFileName, sep='\t', header=False, index=False)

    return train_data_pos, train_data_neg, timeStampFolder_fpga

def detect(detect_bodypart, project_dir, train_data_p, train_data_n, test_annotation_list, vote_sigma, vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, timeStampFolder, dir_keypoints, dir_descriptors):

    print "Performing detections......."
    test_annotations = []

    for test_annotation_file in test_annotation_list:
        test_annotation_file = os.path.join(project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
        with open(test_annotation_file) as fin_annotation:
            test_annotation = json.load(fin_annotation)
            test_annotations.extend(test_annotation["Annotations"])
    print "len(test_annotations):" , len(test_annotations)

    frame_index = -1

    error_stats_all = {}
    n_instance_gt = {}
    n_instance_gt["MouthHook"] = 0
    for s in detect_bodypart:
        error_stats_all[s] = []
        n_instance_gt[s] = 0

    bodypart_gt = {}
    verbosity = 0
    display_level = 3

    bodypart_trained_data_pos = pickle.load(open(train_data_p, 'rb'))
    bodypart_trained_data_neg = pickle.load(open(train_data_n, 'rb'))

    bodypart_knn_pos = FLANN()
    bodypart_knn_pos.build_index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32))
    bodypart_knn_neg = FLANN()
    bodypart_knn_neg.build_index(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float32))

    for j in range(0, len(test_annotations)):
        frame_index += 1
        os.system('clear')
        print "Percentage Complete: %.2f" %(float(frame_index)/float(len(test_annotations))*100)

        annotation = test_annotations[j]

        frame_file = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(project_dir, frame_file)
        frame = cv2.imread(frame_file)

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

        if "MouthHook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
            crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-int(crop_size/2))
            crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-int(crop_size/2))
            # print frame_file

            if (display_level >= 2):
                display_voters = frame.copy()
                display_voters = display_voters[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]
                # display_voters = cv2.cvtColor(display_voters, cv2.COLOR_GRAY2BGR)

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

            ## Read Keypoints
            current_dir = os.path.abspath(os.path.dirname(frame_file))
            parent_dir = os.path.basename(current_dir)
            keypoints_folder = os.path.join(dir_keypoints, parent_dir)
            if not os.path.exists(keypoints_folder):
                print "Folder does not exist !!!"
            keypoints_file = os.path.join(keypoints_folder, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"
            if not os.path.exists(keypoints_file):
                print "%s does not exist !!!" %(keypoints_file)
                continue

            ## Read Descriptors
            current_dir = os.path.abspath(os.path.dirname(frame_file))
            parent_dir = os.path.basename(current_dir)
            descriptors_folder = os.path.join(dir_descriptors, parent_dir)
            if not os.path.exists(descriptors_folder):
                print "Folder does not exist !!!"

            descriptors_file = os.path.join(descriptors_folder, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"
            if not os.path.exists(descriptors_file):
                print "%s does not exist !!!" %(descriptors_file)
                continue

            bodypart_kp, bodypart_desc = readFPGA_KP_DescfromFile(keypoints_file, descriptors_file)

            bodypart_vote_map, numberKP = computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma, bodypart_kp, bodypart_desc, detect_bodypart)

            for bi in range(0, len(detect_bodypart)):
                vote_max = np.amax(bodypart_vote_map[bi][:, :])
                if (vote_max > vote_threshold and ((detect_bodypart[bi] not in bodypart_coords_est) or vote_max >
                    bodypart_coords_est[detect_bodypart[bi]]["conf"])):
                    vote_max_loc = np.array(np.where(bodypart_vote_map[bi][:, :] == vote_max))
                    vote_max_loc = vote_max_loc[:, 0]
                    bodypart_coords_est[detect_bodypart[bi]] = {"conf": vote_max,
                                                                "x": int(vote_max_loc[1]) + int(image_header["crop_x"]),
                                                                "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}

                if (display_level >= 2 and bi == 3):
                    display_vote_map = np.asarray(display_voters, np.float)
                    display_vote_map /= np.amax(display_vote_map)
                    display_vote_map_2 = np.asarray(bodypart_vote_map[bi])
                    display_vote_map_2 /= np.amax(display_vote_map_2)
                    display_vote_map_3 = np.zeros(np.shape(display_vote_map), np.float)
                    display_vote_map_3[:, :, 2] = display_vote_map_2
                    alpha = 0.6
                    display_vote_map = cv2.addWeighted(display_vote_map, alpha, display_vote_map_3, 1 - alpha, 0, display_vote_map)
                    vote_max = np.amax(bodypart_vote_map[bi][:, :])
                    vote_max_loc = np.array(np.where(bodypart_vote_map[bi][:, :] == vote_max))
                    vote_max_loc = vote_max_loc[:, 0]
                    cv2.circle(display_vote_map, (vote_max_loc[1], vote_max_loc[0]), 4, (0, 1, 1), thickness=-1)
                    cv2.imshow("voters", display_vote_map)
                    cv2.waitKey(1000)
        else:
            continue

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

        if ( "detections" in received_json ):
            for di in range(0, len(received_json["detections"])):
                tbp = received_json["detections"][di]["test_bodypart"]
                fi = received_json["detections"][di]["frame_index"]
                if ( tbp in bodypart_gt[fi]["bodypart_coords_gt"] ):
                    error_stats = Error_Stats()
                    error_stats.frame_file =  bodypart_gt[fi]["frame_file"]
                    error_stats.error_distance = np.sqrt(np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"] - received_json["detections"][di]["coord_x"]) + np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"] - received_json["detections"][di]["coord_y"]))
                    error_stats.conf = received_json["detections"][di]["conf"]
                    if (verbosity >= 1 ):
                        print "Frame Index: ",frame_index,"\nDistance between annotated and estimated", tbp, "location:", error_stats.error_distance
                    error_stats_all[tbp].append(error_stats)

    print os.sys.argv
    timestr = time.strftime("%Y%m%d_%H%M%S")

    testList = os.path.join(timeStampFolder, timestr + '_testList.lst')
    listWriter = open(testList, 'w')
    for testFile in test_annotation_list:
        listWriter.write(testFile)
    listWriter.close()

    resultFileName = os.path.join(timeStampFolder, timestr + '_results.log')
    res = open(resultFileName, 'w')
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

    json.dump(testData, res)
    res.close()

if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list-all", dest="train_annotation_list_all", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--test-annotation-list-all", dest="test_annotation_list_all", default="",help="list of frame level training annotation JSON files")
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
    parser.add_option("", "--fpga-dir-kp", dest="fpga_dir_kp",default="",help="Crops surrounding Mouthhook")
    parser.add_option("", "--fpga-dir-desc", dest="fpga_dir_desc",default="",help="Crops surrounding Mouthhook")
    parser.add_option("", "--num-train", dest="num_train",  type="int", default=1, help="Number of training data")

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
    keypt_dir = options.fpga_dir_kp
    desc_dir = options.fpga_dir_desc
    num_train = options.num_train

    crop_size = options.crop_size

    train_list_pos_all = []
    train_list_neg_all = []

    X = []
    with open(train_annotation_list_all) as all_list:
        for train_annotation_list in all_list:
            X.append(train_annotation_list)
    train_annotation_list = X

    combi = 0
    for sub_tl_train in itertools.combinations(train_annotation_list, num_train):
        combi += 1
        sub_tl_test = np.setdiff1d(train_annotation_list, sub_tl_train)
        temp_folder = os.path.join('../expts/', os.path.split(train_annotation_list_all[:-3])[1] + 'grp_' + str(combi))
        timestr = time.strftime("%Y%m%d_%H%M%S")

        # timeStampFolder = os.path.join(temp_folder, timestr)
        timeStampFolder = temp_folder
        if not os.path.exists(timeStampFolder):
            os.makedirs(timeStampFolder)
        temp_train_filename = os.path.join(timeStampFolder, os.path.split(train_annotation_list_all[:-3])[1]  + 'grp_' + str(combi) + '_train.txt')
        temp_test_filename = os.path.join(timeStampFolder, os.path.split(train_annotation_list_all[:-3])[1]  + 'grp_' + str(combi) + '_test.txt')

        with open(temp_train_filename, 'wb') as listWriter:
            for trainFile in sub_tl_train:
                listWriter.write(trainFile)

        with open(temp_test_filename, 'wb') as listWriter:
            for testFile in sub_tl_test:
                listWriter.write(testFile)

        train_data_positive, train_data_negative, timeStampFolder = train(sub_tl_train, project_dir, train_bodypart,
                                                                              hessianThreshold, nOctaves, nOctaveLayers, pos_neg_equal,
                                                                              mh_neighborhood, crop_size, keypt_dir, desc_dir, timeStampFolder)
        readJSON_to_CSV(sub_tl_test,timeStampFolder)

        # detect_bodypart = train_bodypart
        # detect(detect_bodypart, project_dir, train_data_positive, train_data_negative, sub_tl_test, vote_sigma,
        #        vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, timeStampFolder, keypt_dir,
        #        desc_dir)