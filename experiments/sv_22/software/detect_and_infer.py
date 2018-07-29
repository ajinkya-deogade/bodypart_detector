#! /usr/bin/python

import numpy as np
import cv2
import json
from pyflann import *
import re
from optparse import OptionParser
import time
from sklearn import linear_model
from matplotlib import pylab
import pickle
import seaborn as sns
import pandas as pd

global allBodyParts, surf
allBodyParts = ['MouthHook', 'LeftMHhook', 'RightMHhook', 'LeftDorsalOrgan', 'RightDorsalOrgan']
surf = cv2.SURF(250, nOctaves=2, nOctaveLayers=3, extended=0)

class KeyPoint:
   def __init__(self, frame_id, x, y, angle, rel_x, rel_y, bodypart, head_x, head_y):
        self.frame_id = frame_id
        self.pt = (x, y)
        self.angle = angle
        self.rel_pt = (rel_x, rel_y)
        self.bodypart = bodypart
        self.head_pt = (head_x, head_y)

class Error_Stats:
    def __init__(self):
        self.frame_file = None

def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

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

def computeVoteMapOPENCV(bodypart_knn_pos, bodypart_trained_data_pos, frame, vote_patch_size, vote_sigma):

    bodypart_vote_map_op = {}
    for bid in detect_bodypart:
        bodypart_vote_map_op[bid] = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float)

    bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
    for x in range(-vote_patch_size, vote_patch_size + 1):
        for y in range(-vote_patch_size, vote_patch_size + 1):
            bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

    kp_frame, desc_frame = surf.detectAndCompute(frame, None)
    if desc_frame is not None:
        for h, desc in enumerate(desc_frame):
            desc = np.array(desc, np.float32).reshape((1, 64))
            r_pos_all, d_pos_all = bodypart_knn_pos.nn_index(desc, 25, params=dict(checks = 8))
            # r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, params=dict(checks = 8))

            for knn_id in range(0, np.shape(r_pos_all)[1]):
                r_pos = r_pos_all[:,knn_id]
                d_pos = d_pos_all[:,knn_id]
                relative_distance = 0

                if (relative_distance <= desc_distance_threshold):
                    a = np.pi * kp_frame[h].angle / 180.0
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    allBP = bodypart_trained_data_pos.bodypart
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
                            bodypart_vote_map_op[allBP[vi]][y_start:y_end, x_start:x_end] += bodypart_vote
    return bodypart_vote_map_op

def infer(time_lag, bufferPredictedPositions, X_model_ransac, Y_model_ransac, bp):
    buffer_mh = np.vstack(bufferPredictedPositions["MouthHook"])
    buffer_lmh = np.vstack(bufferPredictedPositions["LeftMHhook"])
    buffer_rmh = np.vstack(bufferPredictedPositions["RightMHhook"])
    buffer_ldo = np.vstack(bufferPredictedPositions["LeftDorsalOrgan"])
    buffer_rdo = np.vstack(bufferPredictedPositions["RightDorsalOrgan"])
    X_pred = [-1]
    Y_pred = [-1]

    if bp == "LeftDorsalOrgan":
        bufferAll = np.hstack((buffer_ldo, buffer_mh, buffer_lmh, buffer_rmh))
    elif bp == "RightDorsalOrgan":
        bufferAll = np.hstack((buffer_rdo, buffer_mh, buffer_lmh, buffer_rmh))

    if np.shape(bufferAll)[0] >= time_lag:
        if (X_model_ransac == None and Y_model_ransac == None) and np.shape(bufferAll)[0] == time_lag:
            data_posterior = bufferAll[:time_lag+1]
            X = data_posterior[:, 0]
            Y = data_posterior[:, 1]
            X_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=3, max_trials=100)
            Y_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=3, max_trials=100)
            X_model_ransac.fit(data_posterior, X.reshape(-1, 1))
            Y_model_ransac.fit(data_posterior, Y.reshape(-1, 1))

        elif X_model_ransac != None and Y_model_ransac != None and np.shape(bufferAll)[0] > time_lag:
            data_present_frame = bufferAll[-time_lag-1:-1]
            X_pred = np.squeeze(X_model_ransac.predict(data_present_frame))
            Y_pred = np.squeeze(Y_model_ransac.predict(data_present_frame))

            ## Create model for the next frame
            X_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=3, max_trials=100)
            Y_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=3, max_trials=100)
            data_posterior = bufferAll[-time_lag:]
            X = data_posterior[:, 0]
            Y = data_posterior[:, 1]
            X_model_ransac.fit(data_posterior, X.reshape(-1, 1))
            Y_model_ransac.fit(data_posterior, Y.reshape(-1, 1))

    return X_pred[-1], Y_pred[-1], X_model_ransac, Y_model_ransac

def computeErrors(dat, dat_gt):
    err = {}
    for bp in dat:
        err[bp] = []
        if "x" in dat_gt[bp]:
            err[bp] = np.sqrt(np.power((int(dat[bp]['x']) - int(dat_gt[bp]['x'])), 2) + np.power((int(dat[bp]['y']) - int(dat_gt[bp]['y'])), 2))
    return err

def saveJSON(data, groupString):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    resultFileName = '../expts/' + timestr + '_' + groupString + '_results.log'
    r1 = open(resultFileName, 'w')
    json.dump(data, r1)
    r1.close()
    return resultFileName

def plotErrors(datErr, datString):
    detect_dataframe = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in datErr.iteritems()]))
    print 'Saving figure..............'
    sns.boxplot(data=detect_dataframe)
    sns.swarmplot(data=detect_dataframe, color=".25")
    pylab.ylim(0, 60)
    pylab.ylabel('Distance from ground-truth (pixels)', fontsize=14)
    # pylab.show()
    pylab.savefig('../expts/figures/swarmPlot_' + datString + '.png')
    pylab.clf()

def detect(detect_bodypart, project_dir, test_annotation_list, vote_sigma, vote_patch_size, vote_threshold, train_data_p):
    print "Performing detections......."
    print "Loading training Data......."
    bodypart_trained_data_pos = pickle.load(open(train_data_p, 'rb'))
    print "Creating FLANN Tree........."
    bodypart_knn_pos = FLANN()
    bodypart_knn_pos.build_index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32), params=dict(algorithm=1, trees=4))
    verbosity = 0
    display_level = 0
    time_lag = 6
    infer_bodypart = ["LeftDorsalOrgan", "RightDorsalOrgan"]
    detections_error_all = {}
    inferences_error_all = {}
    show_plot = 0

    for bi in detect_bodypart:
        detections_error_all[bi] = []
        inferences_error_all[bi] = []

    for test_annotation_file in test_annotation_list:
        test_annotation_file = os.path.join(project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
        with open(test_annotation_file) as fin_annotation:
            test_annotation = json.load(fin_annotation)
            test_annotations = test_annotation["Annotations"]
            frame_index = -1

            X_model_ransac = None
            Y_model_ransac = None

            detections = {}
            inferences = {}
            annotations = {}
            detections_error = {}
            inferences_error = {}
            buffer_detection = {}

            for bi in detect_bodypart:
                buffer_detection[bi] = []

            for j in range(0, len(test_annotations)):
                # print "Percentage Complete: %.2f" %(float(frame_index)/float(len(test_annotations))*100)
                frame_index = int(test_annotations[j]["FrameIndexVideo"])
                print frame_index
                annotation = test_annotations[j]

                frame_file = annotation["FrameFile"]
                frame_file = re.sub(".*/data/", "data/", frame_file)
                frame_file = os.path.join(project_dir, frame_file)
                frameOrg = cv2.imread(frame_file)

                detections[frame_index] = {}
                inferences[frame_index] = {}
                annotations[frame_index] = {}

                bodypart_coords_est = {}
                bodypart_coords_inf = {}
                bodypart_coords_gt = {}

                for k in range(0, len(annotation["FrameValueCoordinates"])):
                    bi = annotation["FrameValueCoordinates"][k]["Name"]
                    bodypart_coords_gt[bi] = {"x": int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"]), "y": int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])}
                    if(bi == "MouthHook" or any(bi == s for s in detect_bodypart)):
                        cropCenter_X = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                        cropCenter_Y = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])

                # print annotation["FrameValueCoordinates"]
                # exit()
                crop_x = max(0, cropCenter_X-int(crop_size/2))
                crop_y = max(0, cropCenter_Y-int(crop_size/2))
                frame = frameOrg[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size,0]

                image_info = np.shape(frame)

                if (verbosity >= 2):
                    print image_info

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

                if (display_level >= 2):
                    # display_voters = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    display_voters = cv2.cvtColor(frameOrg, cv2.COLOR_BGR2GRAY)
                    display_voters_2 = cv2.resize(display_voters, (np.shape(display_voters)[0]/4, np.shape(display_voters)[1]/4))
                    cv2.imshow('Frame', display_voters_2)
                    cv2.waitKey(250)

                vote_map = computeVoteMapOPENCV(bodypart_knn_pos, bodypart_trained_data_pos, frame, vote_patch_size, vote_sigma)

                for bi in detect_bodypart:
                    vote_max = np.amax(vote_map[bi][:,:])
                    if (vote_max > vote_threshold):
                        vote_max_loc = np.array(np.where(vote_map[bi][:,:] == vote_max))
                        vote_max_loc = vote_max_loc[:,0]
                        bodypart_coords_est[bi] = {"conf": vote_max, "x": int(vote_max_loc[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                        print "Detected bodypart", bi, "at frame", frame_index
                        buffer_detection[bi].append([int(vote_max_loc[1]) + int(image_header["crop_x"]), int(vote_max_loc[0]) + int(image_header["crop_y"])])
                        if (bi in infer_bodypart):
                           bodypart_coords_inf[bi] = bodypart_coords_est[bi]
                           print "Added the detected bodypart", bi, "to the inference set as well at frame", frame_index
                    else:
                        buffer_detection = {}
                        for bi in detect_bodypart:
                            buffer_detection[bi] = []
                        X_model_ransac = None
                        Y_model_ransac = None

                if len(buffer_detection['MouthHook']) >= time_lag:
                    for ibp in infer_bodypart:
                        if( not ibp in bodypart_coords_inf):
                            print "Inferring for frame", frame_index
                            X_pred, Y_pred, X_model_ransac, Y_model_ransac = infer(time_lag, buffer_detection, X_model_ransac, Y_model_ransac, ibp)
                            bodypart_coords_inf[ibp] = {"x": X_pred, "y": Y_pred}

                detections[frame_index] = bodypart_coords_est
                inferences[frame_index] = bodypart_coords_inf
                annotations[frame_index] = bodypart_coords_gt

                detections_error[frame_index] = computeErrors(bodypart_coords_est, bodypart_coords_gt)
                inferences_error[frame_index] = computeErrors(bodypart_coords_inf, bodypart_coords_gt)

                for bp in detections_error[frame_index]:
                    detections_error_all[bp].append(detections_error[frame_index][bp])

                for bp in inferences_error[frame_index]:
                    inferences_error_all[bp].append(inferences_error[frame_index][bp])

            saveJSON(detections, "Detection")
            saveJSON(inferences, "Inference")
            saveJSON(annotations, "Annotations")
            saveJSON(detections_error, "Detection_Errors")
            saveJSON(inferences_error, "Inference_Errors")
            print "Detection Error Stats:"
            for bp in detect_bodypart:
               print bp, np.min(detections_error_all[bp]), np.median(detections_error_all[bp]), np.mean(detections_error_all[bp]), np.max(detections_error_all[bp])
            print "Inference Error Stats:"
            for bp in infer_bodypart:
               print bp, np.min(inferences_error_all[bp]), np.median(inferences_error_all[bp]), np.mean(inferences_error_all[bp]), np.max(inferences_error_all[bp])

    if show_plot > 0:
        plotErrors(detections_error_all, "Detection_Errors")
        plotErrors(inferences_error_all, "Inference_Errors")

if __name__ == '__main__':
    parser = OptionParser()
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
    train_data_p = options.train_data_pos

    crop_size = options.crop_size

    train_list_pos_all = []
    train_list_neg_all = []
    testListAll = []
    # test_annotation_list = '../config/test_annotation_list'
    test_annotation_list = '../config/quick_test'
    with open(test_annotation_list, 'r') as testList:
        for tlist in testList:
           print tlist 
           testListAll.append(tlist)

    detect_bodypart = train_bodypart

    detect(detect_bodypart, project_dir, testListAll, vote_sigma, vote_patch_size, vote_threshold, train_data_p)
