#! /opt/local/bin/python

import json
import pickle
import random
# import re
import time
from optparse import OptionParser
# from pyflann import *
import os
import cv2
import numpy as np
import csv
# import pandas as pd

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


def readFPGA_KP_DescfromFile(kp_file, desc_file):

    ## Read Keypoints
    keypoints = []
    with open(kp_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
            row = row[0]
            keypoints_temp = cv2.KeyPoint(x=float(row[0]), y=float(row[1]), _size=float((float(row[2])/float(1.2))*float(9)), _angle=float(180*float(row[6])/np.pi), _response=float(row[3]), _octave=int(row[4]), _class_id=int(row[5]))
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

def train(train_annotation_file, project_dir, train_bodypart, hessianThreshold, nOctaves, nOctaveLayers, pos_neg_equal, mh_neighborhood, crop_size, dir_keypoints, dir_descriptors):

    # surf = cv2.xfeatures2d.SURF_create(int(hessianThreshold), nOctaves=int(nOctaves), nOctaveLayers=int(nOctaveLayers), extended=1)

    train_annotation = {}
    train_annotation["Annotations"] = []

    # for train_annotation_file in train_annotation_list:
    #     train_annotation_file = os.path.join(project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
    #     with open(train_annotation_file) as fin_annotation:
    #         tmp_train_annotation = json.load(fin_annotation)
    #         train_annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    bodypart_kp_train_pos = []
    bodypart_desc_train_pos = []
    bodypart_vote_train_pos = []
    bodypart_kp_train_neg = []
    bodypart_desc_train_neg = []
    bodypart_vote_train_neg = []
    training_bodypart = train_bodypart
    keypoints_all = 0

    bodypart_coords_gt_all = []
    with open(train_annotation_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for x, y in reader:
            gt_coord = {}
            gt_coord["x"] = int(float(x))
            gt_coord["y"] = int(float(y))
            bodypart_coords_gt_all.append(gt_coord)

    print 'All Annotations : ', len(bodypart_coords_gt_all)
    mainFileString = os.path.splitext(os.path.basename(train_annotation_file))[0][9:]

    for i in range(0, len(bodypart_coords_gt_all)):
        bodypart_coords = []
        bodypart_coords_gt = {}
        bodypart_kp_train_pos_frame = []
        bodypart_desc_train_pos_frame = []
        bodypart_vote_train_pos_frame = []
        bodypart_kp_train_neg_frame = []
        bodypart_desc_train_neg_frame = []
        bodypart_vote_train_neg_frame = []

        bodypart_coords_gt["x"] = bodypart_coords_gt_all[i]["x"]
        bodypart_coords_gt["y"] = bodypart_coords_gt_all[i]["y"]

        ## Read Keypoints
        keypoints_folder = dir_keypoints
        if not os.path.exists(keypoints_folder):
            print "Folder does not exist !!!"
        keypoints_file = os.path.join(keypoints_folder, 'Rawdata_' + mainFileString + str(i+1)) + ".csv"

        ## Read Descriptors
        descriptors_folder = dir_descriptors
        if not os.path.exists(descriptors_folder):
            print "Folder does not exist !!!"

        descriptors_file = os.path.join(descriptors_folder, 'Rawdata_' + mainFileString + str(i+1)) + ".csv"

        bodypart_kp, bodypart_desc = readFPGA_KP_DescfromFile(keypoints_file, descriptors_file)

        for k in range(0, len(bodypart_kp)):
            keypoints_all = keypoints_all + 1
            x, y = bodypart_kp[k].pt
            a = np.pi * bodypart_kp[k].angle / 180.0

            flag_add_to_pos = False
            votes_tmp = []
            if (np.sqrt(np.square(x - bodypart_coords_gt["x"]) + np.square(y - bodypart_coords_gt["y"])) <= mh_neighborhood):
                flag_add_to_pos = True
                bodypart_dp_pos = np.array([bodypart_coords_gt["x"] - x, bodypart_coords_gt["y"] - y]).T
                bodypart_R_pos = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                bodypart_dp_R_pos = np.dot(bodypart_R_pos, bodypart_dp_pos)
                votes_tmp.append((bodypart_dp_R_pos, training_bodypart))

            if ( flag_add_to_pos ):
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

        os.system('clear')
        print "Training Body Part: ", train_bodypart
        print "Percentage Complete: %.2f" %(float(i)/float(len(bodypart_coords_gt_all))*100)

    print "Number of Positive Training Samples: ", len(bodypart_kp_train_pos)
    print "Number of Negative Training Samples: ", len(bodypart_kp_train_neg)
    print "Number of All Keypoints: ", keypoints_all

    bodypart_desc_train_samples_pos = np.array(bodypart_desc_train_pos)
    bodypart_kp_train_responses_pos = np.arange(len(bodypart_kp_train_pos), dtype=np.float32)
    bodypart_desc_train_samples_neg = np.array(bodypart_desc_train_neg)
    bodypart_kp_train_responses_neg = np.arange(len(bodypart_kp_train_neg), dtype=np.float32)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    timeStampFolder = '../expts/' + timestr
    if not os.path.exists(timeStampFolder):
        os.makedirs(timeStampFolder)

    train_data_pos = os.path.join(timeStampFolder,training_bodypart[0]+'_positive.p')
    train_data_neg = os.path.join(timeStampFolder,training_bodypart[0]+'_negative.p')

    timeStampFolder_FGPA = timeStampFolder + '/FPGA/'
    if not os.path.exists(timeStampFolder_FGPA):
        os.makedirs(timeStampFolder_FGPA)
    trainFileName = os.path.join(timeStampFolder_FGPA, timestr + '_' + training_bodypart[0] + '_trainData.txt')

    dat_all = []
    for i, desc in enumerate(bodypart_desc_train_samples_pos):
        votes = bodypart_vote_train_pos[i]
        dat = []
        dat_local_desc = []
        bp_present = []
        for bp in train_bodypart:
            bp_present.append(0)
            dat_local_desc.append(0)
            dat_local_desc.append(0)
            dat_local_desc.append(0)
        dat_local_desc = np.array(dat_local_desc)
        for vote in votes:
            if vote[1] in train_bodypart:
                pos = train_bodypart.index(vote[1])
                bp_present[pos] = 1
                pos2 = 3*pos
                dat_local_desc[pos2] = 1
                dat_local_desc[pos2+1] = round(vote[0][0], 4)
                dat_local_desc[pos2+2] = round(vote[0][1], 4)

        if np.any(bp_present):
            dat.append(round(sum(desc), 3))
            desc = map(lambda x: round(x, 3), desc)
            dat.extend(desc)
            dat.extend(dat_local_desc)

        dat_all.append(dat)

    dat_all = sorted(dat_all, key=lambda x: x[0])
    with open(trainFileName, 'w') as f:
        writer = csv.writer(f, dialect="excel-tab")
        writer.writerows(dat_all)

    print "Finished Training ........"

    return train_data_pos, train_data_neg

if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list-all", dest="train_annotation_list_all", default="",help="list of frame level training annotation JSON files")
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

    (options, args) = parser.parse_args()

    train_annotation_list_all = options.train_annotation_list_all
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

    crop_size = options.crop_size

    train_list_pos_all = []
    train_list_neg_all = []
    # train_annotation_list_all = '../config/annotation_list_old_new_all_forStratifiedForFPGA_2'
    #    train_annotation_list_all = '../config/annotation_list_old_new_all_forStratified_4'
    train_annotation_file = '../expts/Metadata_20170221_132951_Annotation.txt'
    # train_annotation_file = '../expts/Metadata_20170224_154914_Annotation.csv'

    # train_annotation_list = []
    # with open(train_annotation_list_all) as all_list:
    #     for list in all_list:
    #         train_annotation_list.append(list)

    train_data_positive, train_data_negative = train(train_annotation_file, project_dir, train_bodypart, hessianThreshold, nOctaves, nOctaveLayers, pos_neg_equal, mh_neighborhood, crop_size, keypt_dir, desc_dir)