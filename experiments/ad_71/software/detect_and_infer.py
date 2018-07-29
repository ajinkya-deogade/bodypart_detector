#! /opt/local/bin/python

import numpy as np
import cv2
from struct import *
import json
from pyflann import *
import re
from optparse import OptionParser
import time
import csv
from sklearn import linear_model

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

def readFPGAKP(fname, bodyPart):
    intBytes = 4
    doubleBytes = 8
    fmt_spec = '>'
    int_spec = 'L'
    double_spec = 'd'
    data = {};
    data['HeadPosition'] = []
    data['HeadPosition'].append([])
    data['HeadPosition'].append([])
    data['position'] = []
    data['position'].append([])
    data['position'].append([])
    data['relative_position'] = []
    data['relative_position'].append([])
    data['relative_position'].append([])
    data['frameNumber'] = []
    data['angle'] = []
    with open(fname, 'rb') as f:
        frm = f.read(intBytes)
        frameNumber = unpack_from(fmt_spec+str(1)+int_spec, frm)
        frameNumber = frameNumber[0]
        while frameNumber > 0:
            data['HeadPosition'][0].append(list(unpack_from(fmt_spec+str(1)+double_spec, f.read(1*doubleBytes))))
            data['HeadPosition'][1].append(list(unpack_from(fmt_spec+str(1)+double_spec, f.read(1*doubleBytes))))
            numberKP = unpack_from(fmt_spec+int_spec, f.read(intBytes))
            numberKP = numberKP[0]
            data['frameNumber'].append(frameNumber+1)
            data['position'][0].append(list(unpack_from(fmt_spec+str(numberKP)+double_spec, f.read(numberKP*doubleBytes))))
            data['position'][1].append(list(unpack_from(fmt_spec+str(numberKP)+double_spec, f.read(numberKP*doubleBytes))))
            data['angle'].append(list(unpack_from(fmt_spec+str(numberKP)+double_spec, f.read(numberKP*doubleBytes))))
            data['relative_position'][0].append(list(unpack_from(fmt_spec+str(numberKP)+double_spec, f.read(numberKP*doubleBytes))))
            data['relative_position'][1].append(list(unpack_from(fmt_spec+str(numberKP)+double_spec, f.read(numberKP*doubleBytes))))
            frameNumber = -1
            frm = f.read(intBytes)
            if not frm:
                break
            frameNumber = unpack_from(fmt_spec+str(1)+int_spec, frm)
            frameNumber = frameNumber[0]
    kpData = []
    bodyPart = "MouthHook"
    for i in range(0, len(data['frameNumber'])):
        kpData_frame = []
        for j in range(0, len(data['position'][0][i])):
            kp = KeyPoint(data['frameNumber'][i],data['position'][0][i][j],data['position'][1][i][j],data['angle'][i][j],data['relative_position'][0][i][j],data['relative_position'][1][i][j], bodyPart, data['HeadPosition'][0][i],data['HeadPosition'][1][i])
            kpData_frame.append(kp)
        kpData.append(kpData_frame)

    return kpData

def computeVoteMapOPENCV(bodypart_knn_pos, trainedRelPos, frame, vote_patch_size):
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
            r_pos_all, d_pos_all = bodypart_knn_pos.nn_index(desc, 25, params=dict(checks = 8))
            # r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, params=dict(checks = 8))

            for knn_id in range(0, np.shape(r_pos_all)[1]):
                r_pos = r_pos_all[:,knn_id]
                d_pos = d_pos_all[:,knn_id]
                relative_distance = 0

                if (relative_distance <= desc_distance_threshold):
                    a = np.pi * kp_frame[h].angle / 180.0
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    for vi in allBodyParts:
                        vote_loc = trainedRelPos[r_pos]
                        p = kp_frame[h].pt + np.dot(R, vote_loc)
                        x, y = p
                        if not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size):
                            y_start = int(float(y)) - int(float(vote_patch_size))
                            y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                            x_start = int(float(x)) - int(float(vote_patch_size))
                            x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                            bodypart_vote_map_op[vi][y_start:y_end, x_start:x_end] += bodypart_vote
    return bodypart_vote_map_op

def computeErrorStats(n_instance_gt, error_stats_all, testList):

    testData = {}

    testData["DetectionParameters"] = {}
    # testData["DetectionParameters"]["PositiveTrainFile"] = train_data_positive
    # testData["DetectionParameters"]["NegativeTrainFile"] = train_data_negative
    testData["DetectionParameters"]["TestAnnotationListFile"] = testList
    testData["DetectionParameters"]["VoteSigma"] = vote_sigma
    testData["DetectionParameters"]["VotePatchSize"] = vote_patch_size
    testData["DetectionParameters"]["VoteDescriptorErrorDistance"] = desc_distance_threshold
    testData["DetectionParameters"]["VoteThreshold"] = vote_threshold
    testData["DetectionParameters"]["OutlierErrorDistance"] = outlier_error_dist

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
        print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (outlier_error_dist, n_outlier, n_instance_gt[bid], float(n_outlier) / max(1, float(n_instance_gt[bid])) )

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

    return testData

def computeErrorStatsInference(n_instance_gt, error_stats_all, testList):

    testData = {}

    testData["DetectionParameters"] = {}
    # testData["DetectionParameters"]["PositiveTrainFile"] = train_data_positive
    # testData["DetectionParameters"]["NegativeTrainFile"] = train_data_negative
    testData["DetectionParameters"]["TestAnnotationListFile"] = testList
    testData["DetectionParameters"]["VoteSigma"] = vote_sigma
    testData["DetectionParameters"]["VotePatchSize"] = vote_patch_size
    testData["DetectionParameters"]["VoteDescriptorErrorDistance"] = desc_distance_threshold
    testData["DetectionParameters"]["VoteThreshold"] = vote_threshold
    testData["DetectionParameters"]["OutlierErrorDistance"] = outlier_error_dist

    testData["DetectionResults"] = {}

    for bid in error_stats_all:
        testData["DetectionResults"][bid] = {}
        error_distance_inliers = []
        for es in error_stats_all[bid]:
            if (es.error_distance <= outlier_error_dist):
                error_distance_inliers.append(es.error_distance)

        n_outlier = n_instance_gt[bid] - len(error_distance_inliers)
        testData["DetectionResults"][bid]["NumberInlier"] = len(error_distance_inliers)
        testData["DetectionResults"][bid]["InlierErrorDistance"] = error_distance_inliers

        testData["DetectionResults"][bid]["NumberOutlier"] = n_outlier

        testData["DetectionResults"][bid]["GroundTruthInstance"] = n_instance_gt[bid]
        testData["DetectionResults"][bid]["NumberDetection"] = len(error_stats_all[bid])
        testData["DetectionResults"][bid]["ProportionOutlier"] = float(n_outlier) / max(1, float(n_instance_gt[bid]))

        print "Body part:", bid
        print "Number of inliers: ", len(error_distance_inliers)
        print "Ground truth number of instances:", n_instance_gt[bid]
        print "Total number of detections:", len(error_stats_all[bid])
        print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (outlier_error_dist, n_outlier, n_instance_gt[bid], float(n_outlier) / max(1, float(n_instance_gt[bid])) )

        if (len(error_distance_inliers) > 0):
            testData["DetectionResults"][bid]["MedianInlierErrorDist"] = np.median(error_distance_inliers)
            testData["DetectionResults"][bid]["MeanInlierErrorDist"] = np.mean(error_distance_inliers)

            print "Median inlier error dist =", np.median(error_distance_inliers)
            print "Mean inlier error dist =", np.mean(error_distance_inliers)
        else:
            testData["DetectionResults"][bid]["MedianInlierErrorDist"] = []
            testData["DetectionResults"][bid]["MeanInlierErrorDist"] = []
            testData["DetectionResults"][bid]["MinInlierConfidence"] = []
            testData["DetectionResults"][bid]["MeanInlierConfidence"] = []

    return testData

def saveErrorStats(errData, groupString):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    resultFileName = '../expts/' + timestr + groupString + '_results.log'
    r1 = open(resultFileName, 'w')
    json.dump(errData, r1)
    r1.close()
    return resultFileName

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

def detect(detect_bodypart, project_dir, test_annotation_list, vote_sigma, vote_patch_size, vote_threshold, trainedDesc, trainedRelPos):
    print "Performing detections......."
    test_annotations = []

    for test_annotation_file in test_annotation_list:
        print test_annotation_file
        test_annotation_file = os.path.join(project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
        with open(test_annotation_file) as fin_annotation:
            test_annotation = json.load(fin_annotation)
            test_annotations.extend(test_annotation["Annotations"])
    print "len(test_annotations):" , len(test_annotations)

    frame_index = -1

    error_stats_all_f = {}
    error_stats_all_o = {}
    n_instance_gt = {}
    n_instance_gt["MouthHook"] = 0
    for s in detect_bodypart:
        error_stats_all_f[s] = []
        error_stats_all_o[s] = []
        n_instance_gt[s] = 0

    bodypart_gt = {}
    verbosity = 0
    display_level = 0
    time_lag = 6
    X_model_ransac = None
    Y_model_ransac = None
    infer_bodypart = ["LeftDorsalOrgan"]

    bufferPredictedPositions = {}
    for bi in detect_bodypart:
        bufferPredictedPositions[bi] = []

    for j in range(0, len(test_annotations)):
        frame_index += 1
        # os.system('clear')
        print "Percentage Complete: %.2f" %(float(frame_index)/float(len(test_annotations))*100)

        annotation = test_annotations[j]

        frame_file = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(project_dir, frame_file)
        frameOrg = cv2.imread(frame_file)
        frameIndexVideo = annotation["FrameIndexVideo"]

        if (display_level >= 2):
            display_voters = frameOrg.copy()

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

        bodypart_knn_pos = FLANN()
        bodypart_knn_pos.build_index(np.array(trainedDesc, dtype=np.float32), params=dict(algorithm=1, trees=4))

        if (verbosity >= 2):
            print "bodypart_coords_gt:", bodypart_coords_gt

        if "MouthHook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
            currentIndex = j
            # groundTruthData.append([bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"], bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]])
            crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-int(crop_size/2))
            crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-int(crop_size/2))
            frame = frameOrg[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size,0]

            image_info = np.shape(frame)
            if (verbosity >= 2):
                print image_info

            ack_message_o={}
            bodypart_coords_est_o = {}
            bodypart_coords_inf_o = {}

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
                display_voters = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            voteMapOPENCV = computeVoteMapOPENCV(bodypart_knn_pos, trainedRelPos, frame, vote_patch_size)
        else:
            ## Empty the buffer
            bufferPredictedPositions = {}
            for bi in detect_bodypart:
                bufferPredictedPositions[bi] = []
            X_model_ransac = None
            Y_model_ransac = None
            continue

        for bi in detect_bodypart:
            vote_max_o = np.amax(voteMapOPENCV[bi][:,:])
            print "Vote Max for %s is "%(bi)
            print vote_max_o
            raw_input("Enter")
            if (vote_max_o > vote_threshold and ((bi not in bodypart_coords_est_o) or vote_max_o > bodypart_coords_est_o[bi]["conf"])):
                vote_max_loc_o = np.array(np.where(voteMapOPENCV[bi][:,:] == vote_max_o))
                vote_max_loc_o = vote_max_loc_o[:,0]
                bodypart_coords_est_o[bi] = {"conf": vote_max_o, "x": int(vote_max_loc_o[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc_o[0]) + int(image_header["crop_y"])}
                bufferPredictedPositions[bi].append([int(vote_max_loc_o[1]) + int(image_header["crop_x"]), int(vote_max_loc_o[0]) + int(image_header["crop_y"])])
            else:
                bufferPredictedPositions[bi].append([-1, -1])

        for ibp in infer_bodypart:
            X_pred, Y_pred, X_model_ransac, Y_model_ransac = infer(time_lag, bufferPredictedPositions, X_model_ransac, Y_model_ransac, ibp)
            bodypart_coords_inf_o[ibp] = {"x": X_pred, "y": Y_pred}

        if (display_level >= 2):
            for bi in detect_bodypart:
                displayFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).copy()
                if ("x" in bodypart_coords_est_o[bi]):
                    detection_OPENCV_X = bodypart_coords_est_o[bi]["x"] - int(image_header["crop_x"])
                    detection_OPENCV_Y = bodypart_coords_est_o[bi]["y"] - int(image_header["crop_y"])
                    print "Body part %s, Detection (%d, %d)"%(bi, detection_OPENCV_X, detection_OPENCV_Y)
                    annotation_GT_X = bodypart_gt[frame_index]["bodypart_coords_gt"][bi]["x"] - int(image_header["crop_x"])
                    annotation_GT_Y = bodypart_gt[frame_index]["bodypart_coords_gt"][bi]["y"] - int(image_header["crop_y"])
                    cv2.circle(displayFrame, (detection_OPENCV_X, detection_OPENCV_Y), 4, (0, 0, 255), thickness=-1)
                    # cv2.circle(displayFrame, (annotation_GT_X, annotation_GT_Y), 4, (255, 255, 255), thickness=-1)
                # if bi in infer_bodypart:
                #     if ("x" in bodypart_coords_inf_o[bi]):
                #         inference_OPENCV_X = bodypart_coords_inf_o[bi]["x"] - int(image_header["crop_x"])
                #         inference_OPENCV_Y = bodypart_coords_inf_o[bi]["y"] - int(image_header["crop_y"])
                #         cv2.circle(displayFrame, (inference_OPENCV_X, inference_OPENCV_Y), 4, (0, 255, 0), thickness=-1)
                cv2.imshow("voters", displayFrame)
                cv2.waitKey(500)

        ack_message_o["detections"] = []
        ack_message_o["inference"] = []
        for bi in detect_bodypart:
            if (bi in bodypart_coords_est_o):
                ack_message_o["detections"].append({"frame_index": frame_index,
                                                    "test_bodypart": bi,
                                                    "coord_x": bodypart_coords_est_o[bi]["x"],
                                                    "coord_y": bodypart_coords_est_o[bi]["y"],
                                                    "conf": bodypart_coords_est_o[bi]["conf"]})

        for bi in infer_bodypart:
            if (bi in bodypart_coords_inf_o and (bodypart_coords_inf_o[bi]["x"] != -1)):
                ack_message_o["inference"].append({"frame_index": frame_index,
                                                    "test_bodypart": bi,
                                                    "coord_x": bodypart_coords_inf_o[bi]["x"],
                                                    "coord_y": bodypart_coords_inf_o[bi]["y"]})

        ack_message_o = json.dumps(ack_message_o, separators=(',',':'))
        received_json_o = json.loads(ack_message_o)
        if ("detections" in received_json_o):
            for di in range(0, len(received_json_o["detections"])):
                tbp = received_json_o["detections"][di]["test_bodypart"]
                fi = received_json_o["detections"][di]["frame_index"]
                if (tbp in bodypart_gt[fi]["bodypart_coords_gt"]):
                    error_stats = Error_Stats()
                    error_stats.frame_file = bodypart_gt[fi]["frame_file"]
                    error_stats.error_distance = np.sqrt(np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"] - received_json_o["detections"][di]["coord_x"]) + np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"] - received_json_o["detections"][di]["coord_y"]))
                    error_stats.conf = received_json_o["detections"][di]["conf"]
                    if (verbosity >= 1):
                        print "Frame Index:", frame_index, "\nDistance between annotated and estimated", tbp, "location:", error_stats.error_distance
                    error_stats_all_o[tbp].append(error_stats)

        if ("inference" in received_json_o):
            for di in range(0, len(received_json_o["inference"])):
                tbp = received_json_o["inference"][di]["test_bodypart"]
                fi = received_json_o["inference"][di]["frame_index"]
                if (tbp in bodypart_gt[fi]["bodypart_coords_gt"]):
                    error_stats = Error_Stats()
                    error_stats.frame_file = bodypart_gt[fi]["frame_file"]
                    error_stats.error_distance = np.sqrt(np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"] - received_json_o["inference"][di]["coord_x"]) + np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"] - received_json_o["inference"][di]["coord_y"]))
                    if (verbosity >= 1):
                        print "Frame Index:", frame_index, "\nDistance between annotated and inferred", tbp, "location:", error_stats.error_distance
                    error_stats_all_f[tbp].append(error_stats)

    print ".............Errors for OpenCV Detection......."
    errorData_o = computeErrorStats(n_instance_gt, error_stats_all_o, testList)
    errorData_f = computeErrorStatsInference(n_instance_gt, error_stats_all_f, testList)
    # saveErrorStats(errorData_o, 'opencv')

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

    crop_size = options.crop_size

    train_list_pos_all = []
    train_list_neg_all = []
    testListAll = []
    test_annotation_list = '../config/train_annotation_list_trial'
    with open(test_annotation_list, 'r') as testList:
        for tlist in testList:
            testListAll.append(tlist)

    fN = '../expts/TrainedDataPosMHSortedEdited.txt'

    trainedDesc = []
    trainedRelPos = []
    with open(fN) as f:
        for row in csv.reader(f, delimiter="\t"):
            row = list(row)
            frow = []
            try:
                for item in row:
                    frow.append(float(item))
                trainedDesc.append(list(frow[:128]))
                trainedRelPos.append(list(frow[-2:]))
            except:
                continue

    detect_bodypart = train_bodypart

    detect(detect_bodypart, project_dir, testListAll, vote_sigma, vote_patch_size, vote_threshold, trainedDesc, trainedRelPos)