#! /opt/local/bin/python

import json
import pickle
import re
import time
from optparse import OptionParser
from pyflann import *
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pylab
from sklearn import linear_model
import glob
import os
import csv

global allBodyParts, surf
allBodyParts = ['MouthHook', 'LeftMHhook', 'RightMHhook', 'LeftDorsalOrgan', 'RightDorsalOrgan']
surf = cv2.xfeatures2d.SURF_create(250, nOctaves=2, nOctaveLayers=3, extended=1)

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

def computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma):
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
            desc = np.array(desc, np.float32).reshape((1, 128))
            # r_pos_all, d_pos_all = bodypart_knn_pos.nn_index(desc, 25, params=dict(checks = 8))
            r_pos_all, d_pos_all = bodypart_knn_pos.nn_index(desc, 25, checks=8)
            # r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, params=dict(checks = 8))
            r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, checks=8)

            for knn_id in range(0, np.shape(r_pos_all)[1]):
                r_pos = int(r_pos_all[:,knn_id])
                d_pos = d_pos_all[:,knn_id]
                relative_distance = d_pos - d_neg

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
    return bodypart_vote_map_op, len(kp_frame)

def infer(inference_method, time_lag, bufferPredictedPositions, X_model_ransac, Y_model_ransac, bp):
    if (inference_method == "ransac_track"):
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

    if (inference_method == "simple_translate_rotate"):
        if (bp == "LeftDorsalOrgan"):
            lmh_r1 = [bufferPredictedPositions["LeftMHhook"][-2][0] - bufferPredictedPositions["MouthHook"][-2][0], 
                      bufferPredictedPositions["LeftMHhook"][-2][1] - bufferPredictedPositions["MouthHook"][-2][1]]
            lmh_r2 = [bufferPredictedPositions["LeftMHhook"][-1][0] - bufferPredictedPositions["MouthHook"][-1][0], 
                      bufferPredictedPositions["LeftMHhook"][-1][1] - bufferPredictedPositions["MouthHook"][-1][1]]
            ldo_r1 = [bufferPredictedPositions["LeftDorsalOrgan"][-1][0] - bufferPredictedPositions["MouthHook"][-2][0], 
                      bufferPredictedPositions["LeftDorsalOrgan"][-1][1] - bufferPredictedPositions["MouthHook"][-2][1]]
            cos_theta = np.dot(lmh_r1, ldo_r1)/np.linalg.norm(lmh_r1)/np.linalg.norm(ldo_r1)
            sin_theta = np.cross(lmh_r1, ldo_r1)/np.linalg.norm(lmh_r1)/np.linalg.norm(ldo_r1)
            rotation_mat = np.mat([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            ldo_r2 = rotation_mat * np.mat(lmh_r2).T * np.linalg.norm(ldo_r1) / np.linalg.norm(np.mat(lmh_r2))
            ldo = ldo_r2 + np.mat(bufferPredictedPositions["MouthHook"][-1]).T
            return ldo[0,0], ldo[1,0], X_model_ransac, Y_model_ransac
        if (bp == "RightDorsalOrgan"):
            rmh_r1 = [bufferPredictedPositions["RightMHhook"][-2][0] - bufferPredictedPositions["MouthHook"][-2][0], 
                      bufferPredictedPositions["RightMHhook"][-2][1] - bufferPredictedPositions["MouthHook"][-2][1]]
            rmh_r2 = [bufferPredictedPositions["RightMHhook"][-1][0] - bufferPredictedPositions["MouthHook"][-1][0], 
                      bufferPredictedPositions["RightMHhook"][-1][1] - bufferPredictedPositions["MouthHook"][-1][1]]
            rdo_r1 = [bufferPredictedPositions["RightDorsalOrgan"][-1][0] - bufferPredictedPositions["MouthHook"][-2][0], 
                      bufferPredictedPositions["RightDorsalOrgan"][-1][1] - bufferPredictedPositions["MouthHook"][-2][1]]
            cos_theta = np.dot(rmh_r1, rdo_r1)/np.linalg.norm(rmh_r1)/np.linalg.norm(rdo_r1)
            sin_theta = np.cross(rmh_r1, rdo_r1)/np.linalg.norm(rmh_r1)/np.linalg.norm(rdo_r1)
            rotation_mat = np.mat([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            rdo_r2 = rotation_mat * np.mat(rmh_r2).T * np.linalg.norm(rdo_r1) / np.linalg.norm(np.mat(rmh_r2))
            rdo = rdo_r2 + np.mat(bufferPredictedPositions["MouthHook"][-1]).T
            return rdo[0,0], rdo[1,0], X_model_ransac, Y_model_ransac

def computeErrors(dat, dat_gt):
    err = {}
    for bp in dat_gt:
        err[bp] = []
        if bp in dat:
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
    pylab.ylim(0, 30)
    pylab.ylabel('Distance from ground-truth (pixels)', fontsize=14)
    # pylab.show()
    pylab.savefig('../expts/figures/swarmPlot_' + datString + '.png')
    pylab.clf()

def detect(detect_bodypart, project_dir, test_annotation_list, vote_sigma, vote_patch_size, vote_threshold, train_data_p, train_data_n, outlier_error_dist):
    print "Performing detections......."
    print "Loading training Data......."
    bodypart_trained_data_pos = pickle.load(open(train_data_p, 'rb'))
    bodypart_trained_data_neg = pickle.load(open(train_data_n, 'rb'))
    print "Creating FLANN Tree........."
    bodypart_knn_pos = FLANN()
    # bodypart_knn_pos.build_index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32), params=dict(algorithm=1, trees=4))
    bodypart_knn_pos.build_index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32))
    bodypart_knn_neg = FLANN()
    # bodypart_knn_neg.build_index(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float32), params=dict(algorithm=1, trees=4))
    bodypart_knn_neg.build_index(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float32))

    verbosity = 0
    display_level = 3
    show_plot = 1

    inference_method = "simple_translate_rotate" # "ransac_track"
    time_lag = 2
    infer_bodypart = ["LeftDorsalOrgan", "RightDorsalOrgan"]

    detections_error_all = {}
    inferences_error_all = {}
    detectionsVsinferences_error_all = {}
    det_and_inf_error = {}
    n_annot_gt_total = {}
    continuousAnnotation_all = {}
    continuousDetection_all = {}
    noInference_all = {}
    continuousInference_all = {}
    correctContinuousDetection_all = {}
    correctContinuousInference_all = {}
    histNoContinuousDetection = {}
    histNoContinuousInference = {}
    noDetectionAll = {}
    noDetectionButInferenceAll = {}
    bins = np.arange(0, 6)

    for bi in detect_bodypart:
        detections_error_all[bi] = []
        inferences_error_all[bi] = []
        detectionsVsinferences_error_all[bi] = []
        continuousAnnotation_all[bi] = 0
        continuousDetection_all[bi] = 0
        continuousInference_all[bi] = 0
        noInference_all[bi] = 0
        correctContinuousDetection_all[bi] = 0
        correctContinuousInference_all[bi] = 0
        histNoContinuousDetection[bi] = []
        histNoContinuousInference[bi] = []
        noDetectionButInferenceAll[bi] = []
        noDetectionAll[bi] = []
        for a in range(0, len(bins)):
            noDetectionButInferenceAll[bi].append(0)
            noDetectionAll[bi].append(0)

    for test_annotation_file in test_annotation_list:
        video_file = test_annotation_file
        video_file = re.sub(".*/data/", "data/", video_file)
        video_file = os.path.join(project_dir, video_file)

        head, tail = os.path.split(video_file)
        trackerMetadataFile = os.path.join(project_dir + ',', head + '/trackerMetadata/VotePatchSize_7_VoteSigma_5_Hessian_50_newTraining_onlyDO_revised/' + os.path.splitext(tail)[0], os.path.splitext(tail)[0]) + '_metadata.csv'

        numberFramesTrackerRecorded = 150
        trackerMetaData = np.empty((numberFramesTrackerRecorded, 61), dtype=np.float16)
        trackerMetaData[:] = np.NAN

        rowNum = -1
        with open(trackerMetadataFile, 'rU') as f:
            reader = csv.reader(f)
            for row in reader:
                rowNum += 1
                colNum = -1
                if rowNum > 0:
                    for val in row:
                        colNum += 1
                        try:
                            trackerMetaData[rowNum-1, colNum] = val
                        except:
                            continue
                            print 'Error in value ', val, rowNum-1, colNum

        # print 'Number of Recorded by tracker ', np.shape(trackerMetaData)[0]
        allAnnotationsFolder = os.path.join(project_dir + ',', head + '/Annotations_And_Frames_Continuos')
        annotationFolderPrefix = os.path.splitext(tail)[0][0:-4]
        tempF = allAnnotationsFolder+'/'+annotationFolderPrefix+'_*'
        annotationFolder = glob.glob(tempF)[0]
        # print 'Annotation Folder',  annotationFolder
        annotationFile = glob.glob(annotationFolder + '/*.json')[0]

        with open(annotationFile) as fin_annotationFile:
            temp_annotationData = json.load(fin_annotationFile)
            annotationData = temp_annotationData["Annotations"]

        print "Video File: ", video_file
        cap = cv2.VideoCapture(video_file)

        numberFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # numberFramesTrackerRecorded = np.shape(trackerMetaData)[0]

        X_model_ransac = None
        Y_model_ransac = None

        detections_error = {}
        inferences_error = {}
        buffer_detection = {}
        for bi in detect_bodypart:
                buffer_detection[bi] = []

        allDetectionsMatrix = np.empty((numberFramesTrackerRecorded, 61), dtype=np.float16)
        allDetectionsMatrix[:] = np.NAN

        allAnnotationsMatrix = np.empty((numberFramesTrackerRecorded, 61), dtype=np.float16)
        allAnnotationsMatrix[:] = np.NAN

        frameIndex = -1

        for metaDataIndex in range(0, numberFramesTrackerRecorded):
            if cap.isOpened():
                frameIndex = trackerMetaData[metaDataIndex, 0]
                # print "Frame Index ", frameIndex
                # print 'Frame Index = %d, Metadata Index = %d, NumberTotalFrames = %d' %(frameIndex, metaDataIndex, numberFramesTrackerRecorded)
                cap.set(1, float(frameIndex - 2))
                try:
                    ret, frameOrg = cap.read()
                except:
                    print 'Not able read frame %f' %(float(frameIndex - 2))
                    pass

                headX =  trackerMetaData[metaDataIndex, 8]
                headY =  trackerMetaData[metaDataIndex, 9]
                cropCenter_X = int(headX)
                cropCenter_Y = int(headY)

                bodypart_coords_est = {}
                bodypart_coords_inf = {}
                bodypart_coords_gt = {}
                allDetectionsMatrix[metaDataIndex, 0] = int(frameIndex)
                allAnnotationsMatrix[metaDataIndex, 0] = int(frameIndex)

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

                vote_map, numberKP = computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma)
                # print 'Number of Keypoints Found = %d' %(numberKP)

                ## Gather All the Data into an Array similar to produced by tracker
                bpIndex = 10
                bpIndexVotemax = 43
                for bi in ["MouthHook", "LeftMHhook", "RightMHhook", "LeftDorsalOrgan", "RightDorsalOrgan"]:
                    bpIndex += 2
                    bpIndexVotemax += 1
                    vote_max = np.amax(vote_map[bi][:, :])
                    vote_max_loc = np.array(np.where(vote_map[bi][:,:] == vote_max))
                    vote_max_loc = vote_max_loc[:, 0]
                    bodypart_coords_est[bi] = {"conf": vote_max, "x": int(vote_max_loc[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                    allDetectionsMatrix[metaDataIndex, bpIndex] = bodypart_coords_est[bi]['x']
                    allDetectionsMatrix[metaDataIndex, bpIndex+1] = bodypart_coords_est[bi]['y']
                    allDetectionsMatrix[metaDataIndex, bpIndexVotemax] = bodypart_coords_est[bi]['conf']
                    allDetectionsMatrix[metaDataIndex, 60] = numberKP

                ## Gather All Annotation Data into similar matrix
                for annIndex in range(0, len(annotationData)):
                    sa = int(annotationData[annIndex]['FrameIndexVideo'])
                    if sa == frameIndex:
                        annotations = annotationData[annIndex]['FrameValueCoordinates']
                        bpIndex = 10
                        biIndex = -1
                        for bi in ["MouthHook", "LeftMHhook", "RightMHhook", "LeftDorsalOrgan", "RightDorsalOrgan"]:
                            biIndex += 1
                            if bi == annotations[biIndex]['Name']:
                                bpIndex += 2
                                allAnnotationsMatrix[metaDataIndex, bpIndex] = annotations[biIndex]['Value']['x_coordinate']
                                allAnnotationsMatrix[metaDataIndex, bpIndex+1] = annotations[biIndex]['Value']['y_coordinate']

                for bi in ["MouthHook", "LeftMHhook", "RightMHhook"]:
                    vote_max = np.amax(vote_map[bi][:,:])
                    if (vote_max > vote_threshold):
                        vote_max_loc = np.array(np.where(vote_map[bi][:,:] == vote_max))
                        vote_max_loc = vote_max_loc[:,0]
                        bodypart_coords_est[bi] = {"conf": vote_max, "x": int(vote_max_loc[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                        # print "Was able to detect bodypart", bi
                        # bodypart_coords_inf[bi] = {"x": int(vote_max_loc[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                        buffer_detection[bi].append([int(vote_max_loc[1]) + int(image_header["crop_x"]), int(vote_max_loc[0]) + int(image_header["crop_y"])])
                    else:
                        buffer_detection = {}
                        for bi in detect_bodypart:
                            buffer_detection[bi] = []
                        X_model_ransac = None
                        Y_model_ransac = None

                for bi in ["LeftDorsalOrgan", "RightDorsalOrgan"]:
                    bpIndex += 2
                    bpIndexVotemax += 1
                    vote_max = np.amax(vote_map[bi][:,:])
                    if (vote_max > vote_threshold):
                        vote_max_loc = np.array(np.where(vote_map[bi][:,:] == vote_max))
                        vote_max_loc = vote_max_loc[:,0]
                        bodypart_coords_est[bi] = {"conf": vote_max, "x": int(vote_max_loc[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                        buffer_detection[bi].append([int(vote_max_loc[1]) + int(image_header["crop_x"]),
                                                     int(vote_max_loc[0]) + int(image_header["crop_y"])])
                    else:
                        buffer_detection = {}
                        for bi in detect_bodypart:
                            buffer_detection[bi] = []
                        X_model_ransac = None
                        Y_model_ransac = None

                if (bi in bodypart_coords_inf):
                    buffer_detection[bi].append([int(bodypart_coords_inf[bi]["x"]), int(bodypart_coords_inf[bi]["y"])])

        cap.release()
        cv2.destroyAllWindows()
        pythonDetetionFile = os.path.join(head + '/trackerMetadata/VotePatchSize_7_VoteSigma_5_Hessian_50_newTraining_onlyDO_revised/' + os.path.splitext(tail)[0], os.path.splitext(tail)[0]) + '_python.csv'
        np.savetxt(pythonDetetionFile, allDetectionsMatrix,  delimiter=',')

        annotationFile = os.path.join(head + '/trackerMetadata/VotePatchSize_7_VoteSigma_5_Hessian_50_newTraining_onlyDO_revised/' + os.path.splitext(tail)[0], os.path.splitext(tail)[0]) + '_annotations.csv'
        np.savetxt(annotationFile, allAnnotationsMatrix,  delimiter=',')

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
    train_data_n = options.train_data_neg

    crop_size = options.crop_size
    testListAll = []
    # test_annotation_list = '../config/quick_new_annotations'
    # test_annotation_list = '../config/new_annotations'
    # test_annotation_list = '../config/FPGAValidation_OnlineImplementation_Videos'
    test_annotation_list = '../config/trainAndTestOnSame_205031'

    with open(test_annotation_list, 'rU') as testList:
        for tlist in testList:
            testListAll.append(tlist.strip())

    detect_bodypart = train_bodypart
    print testListAll

    detect(detect_bodypart, project_dir, testListAll, vote_sigma, vote_patch_size, vote_threshold, train_data_p, train_data_n, outlier_error_dist)
