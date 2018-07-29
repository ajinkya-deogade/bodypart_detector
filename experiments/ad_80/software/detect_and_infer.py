#! /opt/local/bin/python

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


def computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_trained_data_neg,
                         frame, vote_patch_size, vote_sigma):
    bodypart_vote_map_op = {}
    for bid in detect_bodypart:
        bodypart_vote_map_op[bid] = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float)

    bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
    for x in range(-vote_patch_size, vote_patch_size + 1):
        for y in range(-vote_patch_size, vote_patch_size + 1):
            bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(
                -0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

    kp_frame, desc_frame = surf.detectAndCompute(frame, None)
    if desc_frame is not None:
        for h, desc in enumerate(desc_frame):
            desc = np.array(desc, np.float32).reshape((1, 128))
            r_pos_all, d_pos_all = bodypart_knn_pos.nn_index(desc, 25, params=dict(checks=8))
            r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, params=dict(checks=8))

            for knn_id in range(0, np.shape(r_pos_all)[1]):
                r_pos = r_pos_all[:, knn_id]
                d_pos = d_pos_all[:, knn_id]
                relative_distance = d_pos - d_neg

                if (relative_distance <= desc_distance_threshold):
                    a = np.pi * kp_frame[h].angle / 180.0
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    allBP = bodypart_trained_data_pos.bodypart
                    for bi in range(0, len(bodypart_trained_data_pos.votes[r_pos])):
                        vote_loc, vote_bodypart = bodypart_trained_data_pos.votes[r_pos][bi]
                        vi = -1
                        for vid in range(0, len(detect_bodypart)):
                            if (detect_bodypart[vid] == vote_bodypart):
                                vi = vid
                                break
                        if (vi == -1):
                            continue
                        p = kp_frame[h].pt + np.dot(R, vote_loc)
                        x, y = p
                        if (not (x <= vote_patch_size or x >= np.shape(frame)[
                            1] - vote_patch_size or y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size)):
                            y_start = int(float(y)) - int(float(vote_patch_size))
                            y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                            x_start = int(float(x)) - int(float(vote_patch_size))
                            x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                            bodypart_vote_map_op[allBP[vi]][y_start:y_end, x_start:x_end] += bodypart_vote
    return bodypart_vote_map_op


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
                data_posterior = bufferAll[:time_lag + 1]
                X = data_posterior[:, 0]
                Y = data_posterior[:, 1]
                X_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0,
                                                              min_samples=3, max_trials=100)
                Y_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0,
                                                              min_samples=3, max_trials=100)
                X_model_ransac.fit(data_posterior, X.reshape(-1, 1))
                Y_model_ransac.fit(data_posterior, Y.reshape(-1, 1))

            elif X_model_ransac != None and Y_model_ransac != None and np.shape(bufferAll)[0] > time_lag:
                data_present_frame = bufferAll[-time_lag - 1:-1]
                X_pred = np.squeeze(X_model_ransac.predict(data_present_frame))
                Y_pred = np.squeeze(Y_model_ransac.predict(data_present_frame))

                ## Create model for the next frame
                X_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0,
                                                              min_samples=3, max_trials=100)
                Y_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0,
                                                              min_samples=3, max_trials=100)
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
            cos_theta = np.dot(lmh_r1, ldo_r1) / np.linalg.norm(lmh_r1) / np.linalg.norm(ldo_r1)
            sin_theta = np.cross(lmh_r1, ldo_r1) / np.linalg.norm(lmh_r1) / np.linalg.norm(ldo_r1)
            rotation_mat = np.mat([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            ldo_r2 = rotation_mat * np.mat(lmh_r2).T * np.linalg.norm(ldo_r1) / np.linalg.norm(np.mat(lmh_r2))
            ldo = ldo_r2 + np.mat(bufferPredictedPositions["MouthHook"][-1]).T
            return ldo[0, 0], ldo[1, 0], X_model_ransac, Y_model_ransac
        if (bp == "RightDorsalOrgan"):
            rmh_r1 = [bufferPredictedPositions["RightMHhook"][-2][0] - bufferPredictedPositions["MouthHook"][-2][0],
                      bufferPredictedPositions["RightMHhook"][-2][1] - bufferPredictedPositions["MouthHook"][-2][1]]
            rmh_r2 = [bufferPredictedPositions["RightMHhook"][-1][0] - bufferPredictedPositions["MouthHook"][-1][0],
                      bufferPredictedPositions["RightMHhook"][-1][1] - bufferPredictedPositions["MouthHook"][-1][1]]
            rdo_r1 = [
                bufferPredictedPositions["RightDorsalOrgan"][-1][0] - bufferPredictedPositions["MouthHook"][-2][0],
                bufferPredictedPositions["RightDorsalOrgan"][-1][1] - bufferPredictedPositions["MouthHook"][-2][1]]
            cos_theta = np.dot(rmh_r1, rdo_r1) / np.linalg.norm(rmh_r1) / np.linalg.norm(rdo_r1)
            sin_theta = np.cross(rmh_r1, rdo_r1) / np.linalg.norm(rmh_r1) / np.linalg.norm(rdo_r1)
            rotation_mat = np.mat([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            rdo_r2 = rotation_mat * np.mat(rmh_r2).T * np.linalg.norm(rdo_r1) / np.linalg.norm(np.mat(rmh_r2))
            rdo = rdo_r2 + np.mat(bufferPredictedPositions["MouthHook"][-1]).T
            return rdo[0, 0], rdo[1, 0], X_model_ransac, Y_model_ransac


def computeErrors(dat, dat_gt):
    err = {}
    for bp in dat_gt:
        err[bp] = []
        if bp in dat:
            if "x" in dat_gt[bp]:
                err[bp] = np.sqrt(np.power((int(dat[bp]['x']) - int(dat_gt[bp]['x'])), 2) + np.power(
                    (int(dat[bp]['y']) - int(dat_gt[bp]['y'])), 2))
    return err


def saveJSON(data, groupString):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    resultFileName = '../expts/' + timestr + '_' + groupString + '_results.log'
    r1 = open(resultFileName, 'w')
    json.dump(data, r1)
    r1.close()
    return resultFileName


def plotErrors(datErr, datString):
    detect_dataframe = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in datErr.iteritems()]))
    print 'Saving figure..............'
    sns.boxplot(data=detect_dataframe)
    sns.swarmplot(data=detect_dataframe, color=".25")
    pylab.ylim(0, 30)
    pylab.ylabel('Distance from ground-truth (pixels)', fontsize=14)
    # pylab.show()
    pylab.savefig('../expts/figures/swarmPlot_' + datString + '.png')
    pylab.clf()


def detect(detect_bodypart, project_dir, test_annotation_list, vote_sigma, vote_patch_size, vote_threshold,
           train_data_p, train_data_n, outlier_error_dist):
    print "Performing detections......."
    print "Loading training Data......."
    bodypart_trained_data_pos = pickle.load(open(train_data_p, 'rb'))
    bodypart_trained_data_neg = pickle.load(open(train_data_n, 'rb'))
    print "Creating FLANN Tree........."
    bodypart_knn_pos = FLANN()
    bodypart_knn_pos.build_index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32),
                                 params=dict(algorithm=1, trees=4))
    bodypart_knn_neg = FLANN()
    bodypart_knn_neg.build_index(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float32),
                                 params=dict(algorithm=1, trees=4))

    verbosity = 0
    display_level = 3
    show_plot = 1

    inference_method = "simple_translate_rotate"  # "ransac_track"
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
        for a in bins:
            noDetectionButInferenceAll[bi].append(0)
            noDetectionAll[bi].append(0)

    fileCount = 0

    for test_annotation_file in test_annotation_list:
        test_annotation_file = os.path.join(project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
        with open(test_annotation_file) as fin_annotation:
            fileCount += 1
            test_annotation = json.load(fin_annotation)
            test_annotations = test_annotation["Annotations"]
            frame_index = -1
            print fileCount

            X_model_ransac = None
            Y_model_ransac = None

            detections = {}
            inferences = {}
            annotations = {}
            detections_error = {}
            inferences_error = {}
            buffer_detection = {}
            annotationExists = {}
            detectionExists = {}
            inferenceExists = {}

            for bi in detect_bodypart:
                buffer_detection[bi] = []
                detectionExists[bi] = []
                inferenceExists[bi] = []
                annotationExists[bi] = []
                inferences_error[bi] = []
                detections_error[bi] = []

            for j in range(0, len(test_annotations)):
                # print "Percentage Complete: %.2f" %((float(j)/float(len(test_annotations)))*100)
                annotation = test_annotations[j]
                # frame_index = int(test_annotations[j]["FrameIndexVideo"])
                # print "frame_index:", frame_index

                frame_file = annotation["FrameFile"]
                frame_file = re.sub(".*/data/", "data/", frame_file)
                frame_file = os.path.join(project_dir, frame_file)
                frameOrg = cv2.imread(frame_file)

                detections[j] = {}
                inferences[j] = {}
                annotations[j] = {}

                bodypart_coords_est = {}
                bodypart_coords_inf = {}
                bodypart_coords_gt = {}

                for bi in detect_bodypart:
                    inferences_error[bi].append([])
                    detections_error[bi].append([])

                for k in range(0, len(annotation["FrameValueCoordinates"])):
                    bi = annotation["FrameValueCoordinates"][k]["Name"]
                    bodypart_coords_gt[bi] = {"x": int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"]),
                                              "y": int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])}
                    if (not bi in n_annot_gt_total):
                        n_annot_gt_total[bi] = 0
                    n_annot_gt_total[bi] += 1

                    if (bi == "MouthHook" or any(bi == s for s in detect_bodypart)):
                        cropCenter_X = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                        cropCenter_Y = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])

                # print annotation["FrameValueCoordinates"]
                # exit()
                crop_x = max(0, cropCenter_X - int(crop_size / 2))
                crop_y = max(0, cropCenter_Y - int(crop_size / 2))
                frame = frameOrg[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, 0]

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

                # if (display_level >= 4):
                #     # display_voters = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                #     display_voters = cv2.cvtColor(frameOrg, cv2.COLOR_BGR2GRAY)
                #     display_voters_2 = cv2.resize(display_voters, (np.shape(display_voters)[0]/4, np.shape(display_voters)[1]/4))
                #     cv2.imshow('Frame', display_voters_2)
                #     cv2.waitKey(250)

                vote_map = computeVoteMapOPENCV(bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos,
                                                bodypart_trained_data_neg, frame, vote_patch_size, vote_sigma)

                for bi in ["MouthHook", "LeftMHhook", "RightMHhook"]:
                    vote_max = np.amax(vote_map[bi][:, :])
                    if (vote_max > vote_threshold):
                        vote_max_loc = np.array(np.where(vote_map[bi][:, :] == vote_max))
                        vote_max_loc = vote_max_loc[:, 0]
                        bodypart_coords_est[bi] = {"conf": vote_max,
                                                   "x": int(vote_max_loc[1]) + int(image_header["crop_x"]),
                                                   "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                        # print "Was able to detect bodypart", bi
                        # bodypart_coords_inf[bi] = {"x": int(vote_max_loc[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                        buffer_detection[bi].append([int(vote_max_loc[1]) + int(image_header["crop_x"]),
                                                     int(vote_max_loc[0]) + int(image_header["crop_y"])])
                    else:
                        buffer_detection = {}
                        for bi in detect_bodypart:
                            buffer_detection[bi] = []
                        X_model_ransac = None
                        Y_model_ransac = None

                if (len(buffer_detection['MouthHook']) >= time_lag and
                            len(buffer_detection['LeftDorsalOrgan']) >= 1 and
                            len(buffer_detection['RightDorsalOrgan']) >= 1):
                    for ibp in infer_bodypart:
                        if not ibp in bodypart_coords_inf:
                            X_pred, Y_pred, X_model_ransac, Y_model_ransac = infer(inference_method, time_lag,
                                                                                   buffer_detection, X_model_ransac,
                                                                                   Y_model_ransac, ibp)
                            bodypart_coords_inf[ibp] = {"x": int(X_pred), "y": int(Y_pred)}

                for bi in ["LeftDorsalOrgan", "RightDorsalOrgan"]:
                    vote_max = np.amax(vote_map[bi][:, :])
                    if (vote_max > vote_threshold):
                        vote_max_loc = np.array(np.where(vote_map[bi][:, :] == vote_max))
                        vote_max_loc = vote_max_loc[:, 0]
                        bodypart_coords_est[bi] = {"conf": vote_max,
                                                   "x": int(vote_max_loc[1]) + int(image_header["crop_x"]),
                                                   "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}
                        # print "Was able to detect bodypart", bi
                        bodypart_coords_inf[bi] = {"x": int(vote_max_loc[1]) + int(image_header["crop_x"]),
                                                   "y": int(vote_max_loc[0]) + int(image_header["crop_y"])}

                    if (bi in bodypart_coords_inf):
                        buffer_detection[bi].append(
                            [int(bodypart_coords_inf[bi]["x"]), int(bodypart_coords_inf[bi]["y"])])

                inferenceExists["LeftDorsalOrgan"].append(0)
                inferenceExists["RightDorsalOrgan"].append(0)

                detectionExists["LeftDorsalOrgan"].append(0)
                detectionExists["RightDorsalOrgan"].append(0)

                annotationExists["LeftDorsalOrgan"].append(0)
                annotationExists["RightDorsalOrgan"].append(0)

                if (display_level >= 3):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # inferredFrame = frameOrg.copy()
                    # folderName = '../expts/annotatedFrames_Inferred/' + os.path.basename(frame_file)[6:-5]
                    if "LeftDorsalOrgan" in bodypart_coords_inf:
                        # cv2.circle(inferredFrame, (bodypart_coords_inf["LeftDorsalOrgan"]['x'], bodypart_coords_inf["LeftDorsalOrgan"]['y']), 10, (0, 255, 255), thickness=-1)
                        inferenceExists["LeftDorsalOrgan"][j] = 1

                    if "RightDorsalOrgan" in bodypart_coords_inf:
                        # cv2.circle(inferredFrame, (bodypart_coords_inf["RightDorsalOrgan"]['x'], bodypart_coords_inf["RightDorsalOrgan"]['y']), 10, (255, 0, 255), thickness=-1)
                        inferenceExists["RightDorsalOrgan"][j] = 1

                    # cv2.putText(inferredFrame, 'Inference', (800, 75), font, 2, (0,0,0), 8)
                    # if not os.path.exists(folderName):
                    #     os.makedirs(folderName)
                    # cv2.imwrite(os.path.join(folderName, os.path.splitext(os.path.basename(frame_file))[0]+'.png'), inferredFrame)

                    # estimatedFrame = frameOrg.copy()
                    # folderName = '../expts/annotatedFrames_Estimated/' + os.path.basename(frame_file)[6:-5]
                    if "LeftDorsalOrgan" in bodypart_coords_est:
                        # cv2.circle(estimatedFrame, (bodypart_coords_est["LeftDorsalOrgan"]['x'], bodypart_coords_est["LeftDorsalOrgan"]['y']), 10, (0, 255, 255), thickness=-1)
                        # cv2.putText(display_voters, '%0.3f'%(bodypart_coords_est["LeftDorsalOrgan"]['conf']), (bodypart_coords_est["LeftDorsalOrgan"]['x']+away, bodypart_coords_est["LeftDorsalOrgan"]['y']+away), font, 2, (0,255,255), 8)
                        # cv2.putText(estimatedFrame, 'Left  DO: %0.2f'%(bodypart_coords_est["LeftDorsalOrgan"]['conf']), (1100, 225), font, 2, (0,255,255), 8)
                        detectionExists["LeftDorsalOrgan"][j] = 1

                    if "RightDorsalOrgan" in bodypart_coords_est:
                        # cv2.circle(estimatedFrame, (bodypart_coords_est["RightDorsalOrgan"]['x'], bodypart_coords_est["RightDorsalOrgan"]['y']), 10, (255, 0, 255), thickness=-1)
                        # cv2.putText(display_voters, '%0.3f'%(bodypart_coords_est["RightDorsalOrgan"]['conf']), (bodypart_coords_est["RightDorsalOrgan"]['x']-away, bodypart_coords_est["RightDorsalOrgan"]['y']-away), font, 2, (255,0,255), 8)
                        # cv2.putText(estimatedFrame, 'Right DO: %0.2f'%(bodypart_coords_est["RightDorsalOrgan"]['conf']), (1100, 300), font, 2, (255,0,255), 8)
                        detectionExists["RightDorsalOrgan"][j] = 1

                    # cv2.putText(estimatedFrame, 'Vote Threshold: %0.2f'%(vote_threshold), (1100, 150), font, 2, (255,255,255), 8)
                    # cv2.putText(estimatedFrame, 'Detection', (800, 75), font, 2, (0,0,0), 8)
                    # if not os.path.exists(folderName):
                    #     os.makedirs(folderName)
                    # cv2.imwrite(os.path.join(folderName, os.path.splitext(os.path.basename(frame_file))[0]+'.png'), estimatedFrame)

                    # gtFrame = frameOrg.copy()
                    # folderName = '../expts/annotatedFrames_GroundTruth/'
                    if bodypart_coords_gt["LeftDorsalOrgan"]['x'] != -1 and bodypart_coords_gt["LeftDorsalOrgan"][
                        'y'] != -1:
                        # cv2.circle(gtFrame, (bodypart_coords_gt["LeftDorsalOrgan"]['x'], bodypart_coords_gt["LeftDorsalOrgan"]['y']), 10, (0, 255, 255), thickness=-1)
                        annotationExists["LeftDorsalOrgan"][j] = 1

                    if bodypart_coords_gt["RightDorsalOrgan"]['x'] != -1 and bodypart_coords_gt["RightDorsalOrgan"][
                        'y'] != -1:
                        # cv2.circle(gtFrame, (bodypart_coords_gt["RightDorsalOrgan"]['x'], bodypart_coords_gt["RightDorsalOrgan"]['y']), 10, (255, 0, 255), thickness=-1)
                        annotationExists["RightDorsalOrgan"][j] = 1

                        # cv2.putText(gtFrame, 'Ground Truth', (800, 75), font, 2, (0,0,0), 8)
                        # if not os.path.exists(folderName):
                        #     os.makedirs(folderName)
                        # cv2.imwrite(os.path.join(folderName, os.path.splitext(os.path.basename(frame_file))[0]+'.png'), gtFrame)

                        # est_inf_Frame = np.concatenate((estimatedFrame, np.multiply(np.ones((1920, 200, 3), dtype=np.uint8), 255), inferredFrame), axis=1)
                        # folderName = '../expts/annotatedFrames_EstimatedWithInferred/' + os.path.basename(frame_file)[6:-5]
                        # if not os.path.exists(folderName):
                        #     os.makedirs(folderName)
                        # cv2.imwrite(os.path.join(folderName, os.path.splitext(os.path.basename(frame_file))[0]+'.png'), est_inf_Frame)

                        # est_inf_gt_Frame = np.concatenate((estimatedFrame, np.multiply(np.ones((1920, 200, 3), dtype=np.uint8), 255), inferredFrame, np.multiply(np.ones((1920, 200, 3), dtype=np.uint8), 255), gtFrame), axis=1)
                        # folderName = '../expts/annotatedFrames_EstimatedWithInferredAndGT/' + os.path.basename(frame_file)[6:-5]
                        # if not os.path.exists(folderName):
                        #     os.makedirs(folderName)
                        # cv2.imwrite(os.path.join(folderName, os.path.splitext(os.path.basename(frame_file))[0]+'.png'), est_inf_gt_Frame)

                        # display_voters_2 = cv2.resize(est_inf_Frame, (np.shape(est_inf_Frame)[0], np.shape(est_inf_Frame)[1]/2))
                        # cv2.imshow('Frame', display_voters_2)
                        # cv2.waitKey(250)

                detections[j] = bodypart_coords_est
                inferences[j] = bodypart_coords_inf
                annotations[j] = bodypart_coords_gt

                for bp in infer_bodypart:
                    if bp in bodypart_coords_gt:
                        if bp in bodypart_coords_est:
                            if bodypart_coords_gt[bp]["x"] != -1 and bodypart_coords_gt[bp]["y"] != -1:
                                detections_error[bp][j] = np.sqrt(
                                    np.power((int(bodypart_coords_est[bp]['x']) - int(bodypart_coords_gt[bp]['x'])),
                                             2) + np.power(
                                        (int(bodypart_coords_est[bp]['y']) - int(bodypart_coords_gt[bp]['y'])), 2))

                        if bp in bodypart_coords_inf:
                            if bodypart_coords_gt[bp]["x"] != -1 and bodypart_coords_gt[bp]["y"] != -1:
                                inferences_error[bp][j] = np.sqrt(
                                    np.power((int(bodypart_coords_inf[bp]['x']) - int(bodypart_coords_gt[bp]['x'])),
                                             2) + np.power(
                                        (int(bodypart_coords_inf[bp]['y']) - int(bodypart_coords_gt[bp]['y'])), 2))

                                # for bp in detections_error[frame_index]:
                                #     detections_error_all[bp].append(detections_error[frame_index][bp])
                                #
                                # for bp in inferences_error[frame_index]:
                                #     inferences_error_all[bp].append(inferences_error[frame_index][bp])

            noDetectionCount = {}
            noDetection = {}
            noDetectionButInference = {}
            noDetectionButInferenceCount = {}
            noDetectionNoInferenceCount = {}

            for bp in infer_bodypart:
                noDetectionCount[bp] = 0
                noDetection[bp] = []
                noDetectionButInference[bp] = []
                noDetectionButInferenceCount[bp] = 0
                noDetectionNoInferenceCount[bp] = 0

                for j in range(1, len(detectionExists[bp]) - 2):
                    # print "GroundTruth Sequence - ", annotationExists[bp][j-1:j+2]
                    # print "Detection Sequence - ", detectionExists[bp][j-1:j+2]
                    # print "Detection Errors - ", detections_error[bp][j-1:j+2]
                    # print "Inference Sequence - ", inferenceExists[bp][j-1:j+2]
                    # print "Inference Errors - ", inferences_error[bp][j-1:j+2]

                    if annotationExists[bp][j - 1] and annotationExists[bp][j] and annotationExists[bp][j + 1]:
                        # if not detectionExists[bp][j] or not (detections_error[bp][j] <= outlier_error_dist):  # More Stricter definition
                        if not detectionExists[bp][j]:
                            noDetectionCount[bp] += 1
                            noDetection[bp].append(1)
                            if inferenceExists[bp][j]:
                                # if inferenceExists[bp][j] and inferences_error[bp][j] <= outlier_error_dist: # More Stricter definition
                                noDetectionButInference[bp].append(1)
                                noDetectionButInferenceCount[bp] += 1
                            else:
                                noDetectionButInference[bp].append(0)
                                noDetectionNoInferenceCount[bp] += 1
                                # print "No Detection Count - ", noDetectionCount[bp]
                                # print "No Detection But Inference Count - ", noDetectionButInferenceCount[bp]
                                # print "No Detection and No Inference Count - ", noDetectionNoInferenceCount[bp]
                        elif noDetectionCount[bp] > 0:
                            # print "There is a detection....."
                            # print "Past detected sequence......", noDetection[bp]
                            # print "Past inferred sequence......", noDetectionButInference[bp]

                            if len(noDetection[bp]) <= len(bins):
                                for jj in range(0, len(noDetection[bp])):
                                    noDetectionAll[bp][jj] += noDetection[bp][jj]
                            else:
                                noDetectionAll[bp][len(bins) - 1] += 1

                            if noDetectionButInference[bp] != []:
                                if len(noDetectionButInference[bp]) < len(bins):
                                    for kk in range(0, len(noDetectionButInference[bp]) - 1):
                                        if noDetectionButInference[bp][kk]:
                                            noDetectionButInferenceAll[bp][kk] += noDetectionButInference[bp][kk]
                                        else:
                                            break
                                else:
                                    noDetectionButInferenceAll[bp][len(bins) - 1] += 1

                            noDetection[bp] = []
                            noDetectionCount[bp] = 0
                            noDetectionButInference[bp] = []
                            noDetectionButInferenceCount[bp] = 0
                            noDetectionNoInferenceCount[bp] = 0
                        else:
                            noDetection[bp].append(0)
                            continue

                    elif annotationExists[bp][j - 1] and annotationExists[bp][j]:
                        # if not detectionExists[bp][j] or not (detections_error[bp][j] <= outlier_error_dist):
                        if not detectionExists[bp][j]:
                            noDetectionCount[bp] += 1
                            noDetection[bp].append(1)
                            # if inferenceExists[bp][j] and inferences_error[bp][j] <= outlier_error_dist:
                            if inferenceExists[bp][j]:
                                noDetectionButInference[bp].append(1)
                                noDetectionButInferenceCount[bp] += 1
                            else:
                                noDetectionButInference[bp].append(0)
                                noDetectionNoInferenceCount[bp] += 1
                                # print "No Detection Count - ", noDetectionCount[bp]
                                # print "No Detection But Inference Count - ", noDetectionButInferenceCount[bp]
                                # print "No Detection and No Inference Count - ", noDetectionNoInferenceCount[bp]
                        elif noDetectionCount[bp] > 0:
                            if len(noDetection[bp]) <= len(bins):
                                for jj in range(0, len(noDetection[bp])):
                                    noDetectionAll[bp][jj] += noDetection[bp][jj]
                            else:
                                noDetectionAll[bp][len(bins) - 1] += 1

                            if noDetectionButInference[bp] != []:
                                if len(noDetectionButInference[bp]) < len(bins):
                                    for kk in range(0, len(noDetectionButInference[bp]) - 1):
                                        if noDetectionButInference[bp][kk]:
                                            noDetectionButInferenceAll[bp][kk] += noDetectionButInference[bp][kk]
                                        else:
                                            break
                                else:
                                    noDetectionButInferenceAll[bp][len(bins) - 1] += 1

                            noDetection[bp] = []
                            noDetectionCount[bp] = 0
                            noDetectionButInference[bp] = []
                            noDetectionButInferenceCount[bp] = 0
                            noDetectionNoInferenceCount[bp] = 0
                        else:
                            noDetection[bp].append(0)
                            continue

                print "noDetectionButInferenceAll - ", noDetectionButInferenceAll[bp]
                print "noDetectionButInferenceAll - ", noDetectionAll[bp]

    for bp in infer_bodypart:
        print "..................Results for Body Part %s .........." % (bp)
        print noDetectionButInferenceAll[bp]
        print noDetectionAll[bp]
        for a in range(0, len(bins)):
            print "Proportion of no correct detections for %d frames leading to correct inference %0.02f" % (
                a, float(noDetectionButInferenceAll[bp][a]) / max(1, float(noDetectionAll[bp][a])))

            # for bp in infer_bodypart:
            #     print "................all stats for body part %s............."%(bp)
            #     print "Number of continuous annotations = %d"%(continuousAnnotation_all[bp])
            #     print "Number of continuous detections = %d"%(continuousDetection_all[bp])
            #     print "Number of correct continuous detections = %d"%(correctContinuousDetection_all[bp])
            #     print "Number of continuous inferences = %d"%(continuousInference_all[bp])
            #     print "Number of correct continuous inferences = %d"%(correctContinuousInference_all[bp])
            #     print "Number of no  continuous inferences = %d"%(noInference_all[bp])
            #     print "Detection Histogram ", np.histogram(histNoContinuousDetection[bp], bins=np.arange(20), density=True)
            #     print "Inference Histogram ", np.histogram(histNoContinuousInference[bp], bins=np.arange(20), density=True)

            # saveJSON(detections, "Detection")
            # saveJSON(inferences, "Inference")
            # saveJSON(annotations, "Annotations")
            # saveJSON(detections_error, "Detection_Errors")
            # saveJSON(inferences_error, "Inference_Errors")

            # for bid in detections_error_all:
            #     print "Error stats for body part:", bid
            #     error_distance_inliers = []
            #     n_stats = 0
            #     for i in range(len(detections_error_all[bid])):
            #         e = detections_error_all[bid][i]
            #         if ( e == [] ):
            #             continue
            #         n_stats += 1
            #         if ( e <= outlier_error_dist):
            #             error_distance_inliers.append(e)

            # print "Ground truth number of instances:", n_annot_gt_total[bid]
            # print "Proportion of inliers (< %d) = %d / %d = %g" % (outlier_error_dist, len(error_distance_inliers), n_annot_gt_total[bid], float(len(error_distance_inliers)) / max(1, float(n_annot_gt_total[bid])))
            # if ( len(error_distance_inliers) > 0 ):
            #     print "Median inlier error dist =", np.median(error_distance_inliers)
            #     print "Mean inlier error dist =", np.mean(error_distance_inliers)

            # if show_plot > 0:
            #     plotErrors(detections_error_all, "Detection_Errors")
            #     plotErrors(inferences_error_all, "Inference_Errors")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="",
                      help="frame level training annotation JSON file")
    parser.add_option("", "--train-annotation-list-all", dest="train_annotation_list_all", default="",
                      help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--mh-neighborhood", dest="mh_neighborhood", type="int", default=100,
                      help="distance from mouth hook for a keyppoint to be considered relevant for training")
    parser.add_option("", "--positive-training-datafile", dest="train_data_pos",
                      help="File to save the information about the positive training data")
    parser.add_option("", "--negative-training-datafile", dest="train_data_neg",
                      help="File to save the information about the negative training data")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",
                      help="display intermediate and final results.write visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--training-bodypart", dest="train_bodypart", default="MouthHook", action="callback",
                      type="string", callback=string_split, help="Input the bodypart to be trained")
    parser.add_option("", "--nOctaves", dest="nOctaves", default=2, type="int",
                      help="Input the number of octaves used in surf object")
    parser.add_option("", "--nOctaveLayers", dest="nOctaveLayers", default=3, type="int",
                      help="Input the number of octave layers used in surf object")
    parser.add_option("", "--hessian-threshold", dest="hessianThreshold", default=250, type="int",
                      help="Input the bodypart to be trained")
    parser.add_option("", "--pos-neg-equal", dest="pos_neg_equal", default=1, type="int",
                      help="Input the bodypart to be trained")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.0,
                      help="threshold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15,
                      help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=5.0,
                      help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--vote-threshold", dest="vote_threshold", type="float", default=0.0,
                      help="threshold on the net vote for a location for it to be a viable detection")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=7,
                      help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--crop-size", dest="crop_size", type="int", default=256, help="Crops surrounding Mouthhook")

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
    # test_annotation_list = '../config/train_annotation_list'
    test_annotation_list = '../config/new_annotations'
    with open(test_annotation_list, 'r') as testList:
        for tlist in testList:
            testListAll.append(tlist)

    detect_bodypart = train_bodypart

    detect(detect_bodypart, project_dir, testListAll, vote_sigma, vote_patch_size, vote_threshold, train_data_p,
           train_data_n, outlier_error_dist)
