#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list", dest="train_annotation_list", default="", help="list of frame level training annotation JSON files")
    parser.add_option("", "--test-annotation", dest="test_annotation_file", default="", help="frame level testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--mh-neighborhood", dest="mh_neighborhood", type="int", default=20, help="distance from mouth hook for a keyppoint to be considered relevant for training")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.1, help="threhsold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15, help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=3.0, help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15, help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int", help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--training-bodypart", dest="train_bodypart",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--save-dir-error", dest="save_dir_error", default="", help="directory to save errors, if at all")

    (options, args) = parser.parse_args()

    print "annotation_file:" , options.train_annotation_file
    error_file = {"num_outliers": 0,"outlier_error_thresh": 0,"proportion_outliers": 0,"median_error_dist": 0,"mean_error_dist": 0,"frames_evaluated": 0}

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            train_annotation = json.load(fin_annotation)
    else:
        train_annotation = {}
        train_annotation["Annotations"] = []
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                train_annotation_file = os.path.join(options.project_dir, re.sub(".*/data/", "data/", train_annotation_file.strip()))
                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    train_annotation["Annotations"].extend(tmp_train_annotation["Annotations"])

    with open(options.test_annotation_file) as fin_annotation:
        test_annotation = json.load(fin_annotation)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    surf = cv2.SURF(400, nOctaves=4, nOctaveLayers=4)

    kp_train = []
    desc_train = []
    vote_train = []
    training_bodypart = options.train_bodypart

    for i in range(0, len(train_annotation["Annotations"])):
        frame_file = train_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir , frame_file)
        print frame_file

        frame = cv2.imread(frame_file)

        if(options.display_level >= 2):
            display_frame = frame.copy()
        else:
            display_frame = None
        
        mh_coords = None
        for j in range(0, len(train_annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                mh_coords = {}
                mh_coords["x"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                mh_coords["y"] = int(train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if (mh_coords is not None):
            kp, desc = surf.detectAndCompute(frame, None)
            for k in range(0, len(kp)):
                x,y = kp[k].pt
                a = np.pi * kp[k].angle/180.0
                # if key point is less than certain distance from the 
                if(np.sqrt(np.square(x - mh_coords["x"]) + np.square(y - mh_coords["y"])) <= options.mh_neighborhood):
                    kp_train.append(kp[k])
                    desc_train.append(desc[k])
                    dp = np.array([mh_coords["x"] - x, mh_coords["y"] - y]).T
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
                    dp_R = np.dot(R, dp)
                    vote_train.append(dp_R)
            
            if(display_frame != None):
                cv2.circle(display_frame, (mh_coords["x"], mh_coords["y"]), 4, (0, 0, 255), thickness=-1)
                cv2.circle(display_frame, (mh_coords["x"], mh_coords["y"]), options.mh_neighborhood, (0, 255, 255), thickness=3)

        if(display_frame != None):
            display_frame = cv2.resize(display_frame, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("mouth hook annotation", display_frame)

    print len(kp_train)
    knn = cv2.KNearest()
    desc_train_samples = np.array(desc_train)
    kp_train_responses = np.arange(len(kp_train),dtype = np.float32)
    knn.train(desc_train_samples,kp_train_responses)

    vote = np.zeros((2*options.vote_patch_size+1, 2*options.vote_patch_size+1, 1), np.float)
    for x in range(-options.vote_patch_size, options.vote_patch_size+1):
        for y in range(-options.vote_patch_size, options.vote_patch_size+1):
            vote[y+options.vote_patch_size,x+options.vote_patch_size] = 1.0 + np.exp(-0.5*(x*x+y*y)/(np.square(options.vote_sigma))) / (options.vote_sigma*np.sqrt(2*np.pi))

    error_dists = []
    n_outlier_error_dist = 0
    n_eval = 0
    print "Training....."
            
    for i in range(0, len(test_annotation["Annotations"])):
        frame_file = test_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir , frame_file)
        #print frame_file

        frame = cv2.imread(frame_file)

        if(options.display_level >= 2):
            display_voters = frame.copy()
        
        mh_coords_gt = None
        for j in range(0, len(test_annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == training_bodypart and test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                mh_coords_gt = {}
                mh_coords_gt["x"] = int(test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                mh_coords_gt["y"] = int(test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        vote_map = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 1), np.float)

        kp_frame, desc_frame = surf.detectAndCompute(frame, None)
        for h, desc in enumerate(desc_frame):
            desc = np.array(desc,np.float32).reshape((1, 128))
            retval, results, neigh_resp, dists = knn.find_nearest(desc, 1)
            r,d =  int(results[0][0]),dists[0][0]
            if (d <= options.desc_distance_threshold):
                a = np.pi * kp_frame[h].angle / 180.0
                R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                p = kp_frame[h].pt + np.dot(R, vote_train[r])
                x,y = p
                if (not(x <= options.vote_patch_size or x >= np.shape(frame)[1]-options.vote_patch_size or y <= options.vote_patch_size or y >= np.shape(frame)[0]-options.vote_patch_size)):
                    vote_map[y-options.vote_patch_size:y+options.vote_patch_size+1, x-options.vote_patch_size:x+options.vote_patch_size+1] += vote
                    if (options.display_level >= 2):
                        cv2.circle(display_voters, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)

        if (options.display_level >= 2):
            display_voters = cv2.resize(display_voters, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("voters", display_voters)

        vote_max = np.amax(vote_map)
        vote_map /= vote_max
        vote_max_loc = np.where(vote_map == np.amax(vote_map))
        mh_coords_est = {}
        mh_coords_est["x"] = int(vote_max_loc[1])
        mh_coords_est["y"] = int(vote_max_loc[0])
        if (mh_coords_gt is not None):
            error_dist = np.sqrt(np.square(mh_coords_gt["x"] - mh_coords_est["x"]) + np.square(mh_coords_gt["y"] - mh_coords_est["y"]))
            #print "Distance between annotated and estimated MH location:", error_dist
            n_eval += 1
            if (error_dist <= options.outlier_error_dist):
                error_dists.append(error_dist)
            else:
                n_outlier_error_dist += 1
        
        if (options.display_level >= 1):
            display_vote_map = np.array(frame.copy(), np.float)
            display_vote_map /= 255.0
            display_vote_map[:,:,2] = vote_map[:,:,0]
            cv2.circle(display_vote_map, (mh_coords_est["x"], mh_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
            display_vote_map = cv2.resize(display_vote_map, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("voters", display_vote_map)

        if (options.display_level >= 1):
            key_press = cv2.waitKey(-1)
            if (key_press == 113 or key_press == 13):
                break

    if (options.save_dir_error != ""):
       error_file = {"num_outliers": n_outlier_error_dist,"outlier_error_thresh": options.outlier_error_dist,"proportion_outliers": float(n_outlier_error_dist) / float(n_eval),"median_error_dist": np.median(error_dists),"mean_error_dist": np.mean(error_dists),"frames_evaluated": n_eval}
       with open(options.save_dir_error,'w') as fout_error:
            json.dump(error_file, fout_error, indent=4)

    print os.sys.argv
    print "Number of outlier error distances (beyond %d) = %d / %d = %g" % ( options.outlier_error_dist, n_outlier_error_dist, n_eval, float(n_outlier_error_dist) / float(n_eval) )
    print "Median inlier error dist =", np.median(error_dists)
    print "Mean inlier error dist =", np.mean(error_dists)

    cv2.destroyWindow("frame")
