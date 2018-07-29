#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle
from multiprocessing import Pool
from multiprocessing import Manager
from threading import Lock
from pyflann import *

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

class Error_Stats:
    def __init__(self):
        self.frame_file = None
        self.error_distance = None

surf = cv2.xfeatures2d.SURF_create(150, nOctaves=2, nOctaveLayers=3, extended=1)

bodypart_knn_pos = FLANN()
# bodypart_knn_neg = FLANN()


bodypart_trained_data_pos = None

lock = Lock()
manager = Manager()
error_stats_all = manager.list([])

options = None
test_bodypart = None

def detect_bodypart_and_get_error_distances(annotation):
    global options
    global test_bodypart
    global error_stats_all
    global bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos

    bodypart_error_dists = []
    bodypart_n_outlier_error_dist = 0
    bodypart_n_eval = 0
    bodypart_vote = np.zeros((2 * options.vote_patch_size + 1, 2 * options.vote_patch_size + 1, 1), np.float)
    
    for x in range(-options.vote_patch_size, options.vote_patch_size + 1):
        for y in range(-options.vote_patch_size, options.vote_patch_size + 1):
            bodypart_vote[y + options.vote_patch_size, x + options.vote_patch_size] = 1.0 + np.exp(
                -0.5 * (x * x + y * y) / (np.square(options.vote_sigma))) / (options.vote_sigma * np.sqrt(2 * np.pi))
            
    frame_file = annotation["FrameFile"]
    frame_file = re.sub(".*/data/", "data/", frame_file)
    frame_file = os.path.join(options.project_dir, frame_file)

    frame = cv2.imread(frame_file)
    
    if (options.display_level >= 2):
        display_voters = frame.copy()
            
    bodypart_coords_gt = None
        
    for j in range(0, len(annotation["FrameValueCoordinates"])):
        if (annotation["FrameValueCoordinates"][j]["Name"] == test_bodypart and annotation["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
            bodypart_coords_gt = {}
            bodypart_coords_gt["x"] = int(annotation["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
            bodypart_coords_gt["y"] = int(annotation["FrameValueCoordinates"][j]["Value"]["y_coordinate"])
        
    bodypart_vote_map = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 1), np.float)
    vote_patch_size = options.vote_patch_size

    kp_frame, desc_frame = surf.detectAndCompute(frame, None)
    for h, desc in enumerate(desc_frame):
        desc = np.array(desc, np.float64).reshape((1, 128))
        r_pos, d_pos = bodypart_knn_pos.nn_index(desc, 1, params=dict(checks=8))
        # r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, params=dict(checks=8))
        relative_distance = 0
        # raw_input("Enterr.....")
        if (relative_distance <= options.desc_distance_threshold):
            a = np.pi * kp_frame[h].angle / 180.0
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            for bi in range(0, len(bodypart_trained_data_pos.votes[r_pos])):
                vote_loc, vote_bodypart = bodypart_trained_data_pos.votes[r_pos][bi]
                print vote_bodypart
                vi = -1
                for vid in range(0, len(options.detect_bodypart)):
                    if (options.detect_bodypart[vid] == vote_bodypart):
                        vi = vid
                        break
                if(vi == -1):
                    continue
                p = kp_frame[h].pt + np.dot(R, vote_loc)
                x, y = p

            if (not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or  y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size)):
                y_start = int(float(y)) - int(float(vote_patch_size))
                y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                x_start = int(float(x)) - int(float(vote_patch_size))
                x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                bodypart_vote_map[y_start:y_end, x_start:x_end] += bodypart_vote

            if (options.display_level >= 2):
                cv2.circle(display_voters, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)

    if (options.display_level >= 2):
        display_voters = cv2.resize(display_voters, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("voters", display_voters)

    vote_max = np.amax(bodypart_vote_map)
    bodypart_vote_map /= vote_max
    vote_max_loc = np.array(np.where(bodypart_vote_map == np.amax(bodypart_vote_map)))
    vote_max_loc = vote_max_loc[:,0]
    bodypart_coords_est = {}
    bodypart_coords_est["x"] = int(vote_max_loc[1])
    bodypart_coords_est["y"] = int(vote_max_loc[0])

    if (bodypart_coords_gt is not None):
        error_stats = Error_Stats()
        error_stats.frame_file = frame_file
        error_stats.error_distance = np.sqrt(np.square(bodypart_coords_gt["x"] - bodypart_coords_est["x"]) + 
                                             np.square(bodypart_coords_gt["y"] - bodypart_coords_est["y"]))
        print frame_file, "Distance between annotated and estimated RightMHhook location:", error_stats.error_distance
            
        # lock.acquire()
        error_stats_all.append(error_stats)
        # lock.release()

    if (options.display_level >= 1):
        display_vote_map = np.array(frame.copy(), np.float)
        display_vote_map /= 255.0
        display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
        cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
        display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("voters", display_vote_map)

    if (options.display_level >= 1):
        key_press = cv2.waitKey(-1)

    return True

def main(options, args):
    global test_bodypart
    global error_stats_all
    global bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos

    print "Reading pickled data........"
    # bodypart_trained_data_pos = SaveClass()
    bodypart_trained_data_pos = pickle.load(open(options.train_data_p, 'rb'))
    # bodypart_trained_data_neg = SaveClass()
    bodypart_trained_data_neg = pickle.load(open(options.train_data_n, 'rb'))

    test_bodypart = bodypart_trained_data_neg.bodypart
    print "test_bodypart:" , test_bodypart

    print "Building KNN Tree....."
    bodypart_knn_pos.build_index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float64), params=dict(algorithm=1, trees=8))
    # bodypart_knn_neg.build_index(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float64), params=dict(algorithm=1, trees=8))

    test_annotations = []
    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.extend(test_annotation["Annotations"])

    print "len(test_annotations):", len(test_annotations)
                
    process_pool = Pool(processes=options.n_thread)
    result = process_pool.map(detect_bodypart_and_get_error_distances, test_annotations)
    # lop = 0
    # for test_annotation in test_annotations:
    #     lop += 1
    #     print "Annotation: ", lop
    #     result = detect_bodypart_and_get_error_distances(test_annotation)
    process_pool.close()
    process_pool.join()
    
    print os.sys.argv
    error_distance_inliers = []
    for es in error_stats_all:
        if (es.error_distance <= options.outlier_error_dist):
            error_distance_inliers.append(es.error_distance)
    n_outlier = len(error_stats_all) - len(error_distance_inliers)
    print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (options.outlier_error_dist, n_outlier, len(error_stats_all), float(n_outlier) / max(1, float(len(error_stats_all))))
    print "Median inlier error dist =", np.median(error_distance_inliers)
    print "Mean inlier error dist =", np.mean(error_distance_inliers)

    cv2.destroyWindow("frame")

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation", dest="test_annotation_file", default="",help="frame level testing annotation JSON file")
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--positive-training-datafile", dest="train_data_p", help="File to save the information about the positive training data")
    parser.add_option("", "--negative-training-datafile", dest="train_data_n", help="File to save the information about the negative training data")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.1,help="threshold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15,help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=3.0,help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--nthread", dest="n_thread", type="int", default=1, help="maximum number of threads for multiprocessing")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--detect-bodypart", dest="detect_bodypart", default="", help="directory to save result visualizations, if at all")

    (options, args) = parser.parse_args()

    main(options, args)