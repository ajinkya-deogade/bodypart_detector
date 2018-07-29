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
import glob

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
    for i in range(0, len(data['frameNumber'])):
        kpData_frame = []
        for j in range(0, len(data['position'][0][i])):
            kp = KeyPoint(data['frameNumber'][i],data['position'][0][i][j],data['position'][1][i][j],data['angle'][i][j],data['relative_position'][0][i][j],data['relative_position'][1][i][j], bodyPart, data['HeadPosition'][0][i],data['HeadPosition'][1][i])
            kpData_frame.append(kp)
        kpData.append(kpData_frame)

    return kpData

def computeVoteMapFPGA(bodypartVoteMap, bodypart_vote, keypointData, frame, vote_patch_size):
    for knn_id in range(0, len(keypointData)):
        relative_distance = 0
        if (relative_distance <= desc_distance_threshold):
            kp = keypointData[knn_id]
            a = (kp.angle*3.14)
            if a < 0:
                a += 6.28
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

            for bi in allBodyParts:
                if bi == kp.bodypart:
                    vote_loc = kp.rel_pt
                    kpy, kpx = kp.pt
                    dotP = np.dot(R, vote_loc)
                    x = kpy + dotP[0]
                    y = kpx + dotP[1]
                    if not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size):
                        y_start = int(float(y)) - int(float(vote_patch_size))
                        y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                        x_start = int(float(x)) - int(float(vote_patch_size))
                        x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                        bodypartVoteMap[bi][y_start:y_end, x_start:x_end] += bodypart_vote

    return bodypartVoteMap

def computeVoteMapOPENCV(bodypart_vote_map, bodypart_vote, bodypart_knn_pos, trainedRelPos, frame, vote_patch_size):
    kp_frame, desc_frame = surf.detectAndCompute(frame, None)
    if desc_frame is not None:
        for h, desc in enumerate(desc_frame):
            desc = np.array(desc, np.float64).reshape((1, 128))
            r_pos_all, d_pos_all = bodypart_knn_pos.nn_index(desc, 25, params=dict(checks=8))
            # r_neg, d_neg = bodypart_knn_neg.nn_index(desc, 1, params=dict(checks = 8))

            for knn_id in range(0, np.shape(r_pos_all)[1]):
                r_pos = r_pos_all[:,knn_id]
                d_pos = d_pos_all[:,knn_id]
                relative_distance = 0

                if (relative_distance <= desc_distance_threshold):
                    a = np.pi * kp_frame[h].angle / 180.0
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    for vi in allBodyParts:
                        if vi == bodypart:
                            vote_loc = trainedRelPos[r_pos]
                            p = kp_frame[h].pt + np.dot(R, vote_loc)
                            x, y = p
                            if not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size):
                                y_start = int(float(y)) - int(float(vote_patch_size))
                                y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                                x_start = int(float(x)) - int(float(vote_patch_size))
                                x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                                bodypart_vote_map[vi][y_start:y_end, x_start:x_end] += bodypart_vote
    return bodypart_vote_map

def computeErrorStats(n_instance_gt, error_stats_all, testListFile):

    testData = {}

    testData["DetectionParameters"] = {}
    testData["DetectionParameters"]["TestAnnotationListFile"] = testListFile
    testData["DetectionParameters"]["VoteSigma"] = vote_sigma
    testData["DetectionParameters"]["VotePatchSize"] = vote_patch_size
    testData["DetectionParameters"]["VoteDescriptorErrorDistance"] = desc_distance_threshold
    testData["DetectionParameters"]["VoteThreshold"] = vote_threshold
    testData["DetectionParameters"]["OutlierErrorDistance"] = outlier_error_dist

    testData["DetectionResults"] = {}
    for bid in error_stats_all:
        testData["DetectionResults"][bid] = {}
        testData["DetectionResults"][bid]["Annotation"] = {}
        testData["DetectionResults"][bid]["Detection"] = {}
        testData["DetectionResults"][bid]["Annotation"]["x"] = []
        testData["DetectionResults"][bid]["Annotation"]["y"] = []
        testData["DetectionResults"][bid]["Detection"]["x"] = []
        testData["DetectionResults"][bid]["Detection"]["y"] = []

    for bid in error_stats_all:
        error_distance_inliers = []
        inlier_confs = []
        outlier_confs = []
        for es in error_stats_all[bid]:
            testData["DetectionResults"][bid]["Annotation"]["x"].append(es.annotation_x)
            testData["DetectionResults"][bid]["Annotation"]["y"].append(es.annotation_y)
            testData["DetectionResults"][bid]["Detection"]["x"].append(es.detection_x)
            testData["DetectionResults"][bid]["Detection"]["y"].append(es.detection_y)
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
            testData["DetectionResults"][bid]["MeanOutlierConfidence"] = np.mean(outlier_confs)

            print "Max outlier confidence = ", np.max(outlier_confs)
            print "Mean outlier confidence = ", np.mean(outlier_confs)
        else:
            testData["DetectionResults"][bid]["MaxOutlierConfidence"] = []
            testData["DetectionResults"][bid]["MeanOutlierConfidence"]  = []

    return testData

def saveErrorStats(errData, groupString):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    resFN = '../expts/' + timestr + '_' + bodypart + '_' + groupString + '_results.log'
    with open(resFN, 'w') as resWrite:
        json.dump(errData, resWrite)
    return resFN

def detect(detect_bodypart, project_dir, knn_dir, test_annotation_list, vote_sigma, vote_patch_size, vote_threshold, bodypart, trainedDesc, trainedRelPos):
    print "Performing detections for bodypart %s" %(bodypart)
    test_annotations = []
    kpDataAll = []

    for test_annotation_file in test_annotation_list:
        # print test_annotation_file
        test_annotation_file = os.path.join(project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
        with open(test_annotation_file) as fin_annotation:
            test_annotation = json.load(fin_annotation)
            # test_annotations.extend(test_annotation["Annotations"])
            test_annotations.append(test_annotation["Annotations"])
            videoFileName = test_annotation["VideoFile"]
            kp_file = re.sub(".*/Clips/", "", videoFileName)
            kp_file = re.sub(".mp4", "", kp_file)
            knnFilePath = os.path.join(knn_dir, kp_file, bodypart, 'NN_KP*.bin')
            kp_file = glob.glob(knnFilePath)
            kdat = readFPGAKP(kp_file[0], bodypart)
            kpDataAll.append(kdat)
    print "len(test_annotations):", len(test_annotations)

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
    for ta in range(0, len(test_annotations)):
        kpData = kpDataAll[ta]
        for j in range(0, len(test_annotations[ta])):
            frame_index += 1
            os.system('clear')
            print "Percentage Complete: %.2f" %(float(frame_index)/float(len(test_annotations)*len(test_annotations[ta]))*100)

            annotation = test_annotations[ta][j]
            frame_file = annotation["FrameFile"]
            frame_file = re.sub(".*/data/", "data/", frame_file)
            frame_file = os.path.join(project_dir, frame_file)
            frameOrg = cv2.imread(frame_file)
            frameIndexVideo = annotation["FrameIndexVideo"]

            if (display_level >= 2):
                display_voters = frameOrg.copy()

            keypointData_pre = kpData[int(frameIndexVideo) - 1]
            keypointData = kpData[int(frameIndexVideo) - 1]

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
            bodypart_knn_pos.build_index(np.array(trainedDesc, dtype=np.float64), params=dict(algorithm=1, trees=8))

            if (verbosity >= 2):
                print "bodypart_coords_gt:", bodypart_coords_gt

            if "MouthHook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
                crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-int(crop_size/2))
                crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-int(crop_size/2))
                frame = frameOrg[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size,0]

                offsetCenters_FPGA_OpenCV = {}
                # offsetCenters_FPGA_OpenCV['x'] = crop_x - (int(keypointData_pre[0].head_pt[0][0] - 128))
                # offsetCenters_FPGA_OpenCV['y'] = crop_y - (int(keypointData_pre[0].head_pt[1][0] - 128))
                offsetCenters_FPGA_OpenCV['x'] = crop_x - (int(keypointData_pre[0].head_pt[0][0]))
                offsetCenters_FPGA_OpenCV['y'] = crop_y - (int(keypointData_pre[0].head_pt[1][0]))

                bodypart_vote_map_fp = {}
                for bid in allBodyParts:
                    bodypart_vote_map_fp[bid] = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float)

                bodypart_vote_map_op = {}
                for bid in allBodyParts:
                    bodypart_vote_map_op[bid] = np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float)

                bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
                for x in range(-vote_patch_size, vote_patch_size + 1):
                    for y in range(-vote_patch_size, vote_patch_size + 1):
                        bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

                image_info = np.shape(frame)

                if (verbosity >= 2):
                    print image_info

                if (verbosity >= 2):
                    print "Bodypart Vote: ", np.shape(bodypart_vote)

                ack_message_f={}
                ack_message_o={}
                bodypart_coords_est_f = {}
                bodypart_coords_est_o = {}

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

                voteMapFPGA = computeVoteMapFPGA(bodypart_vote_map_fp, bodypart_vote, keypointData, frame, vote_patch_size)
                voteMapOPENCV = computeVoteMapOPENCV(bodypart_vote_map_op, bodypart_vote, bodypart_knn_pos, trainedRelPos, frame, vote_patch_size)
            else:
                continue

            for bp in allBodyParts:
                vote_max_f = np.amax(voteMapFPGA[bp][:,:])
                vote_max_o = np.amax(voteMapOPENCV[bp][:,:])
                if (vote_max_f > vote_threshold and ((bp not in bodypart_coords_est_f) or vote_max_f > bodypart_coords_est_f[bp]["conf"])) and (vote_max_o > vote_threshold and ((bp not in bodypart_coords_est_o) or vote_max_o > bodypart_coords_est_o[bp]["conf"])):
                    vote_max_loc_f = np.array(np.where(voteMapFPGA[bp][:,:] == vote_max_f))
                    vote_max_loc_o = np.array(np.where(voteMapOPENCV[bp][:,:] == vote_max_o))
                    vote_max_loc_f = vote_max_loc_f[:,0]
                    vote_max_loc_o = vote_max_loc_o[:,0]
                    bodypart_coords_est_f[bp] = {"conf" : vote_max_f, "x": int(vote_max_loc_f[1]) + int(image_header["crop_x"]) - int(offsetCenters_FPGA_OpenCV['x']), "y": int(vote_max_loc_f[0]) + int(image_header["crop_y"]) - int(offsetCenters_FPGA_OpenCV['y'])}
                    bodypart_coords_est_o[bp] = {"conf" : vote_max_o, "x": int(vote_max_loc_o[1]) + int(image_header["crop_x"]), "y": int(vote_max_loc_o[0]) + int(image_header["crop_y"])}
                    displayFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).copy()
                    if (display_level >= 2):
                        if ("x" in bodypart_coords_est_f[detect_bodypart[bi]] and "x" in bodypart_coords_est_o[detect_bodypart[bi]]):
                            # for kk in keypointData:
                                # cv2.circle(displayFrame, (int(kk.pt[0])-int(image_header["crop_x"]), int(kk.pt[1])-int(image_header["crop_y"])), 4, (0, 255, 255), thickness=-1)
                                # cv2.circle(displayFrame, (int(kk.pt[0]), int(kk.pt[1])), 4, (255, 0, 0), thickness=-1)
                                # cv2.circle(displayFrame, (int(kk.pt[0]) - offsetCenters_FPGA_OpenCV['x'], int(kk.pt[1]) - offsetCenters_FPGA_OpenCV['y']), 4, (0, 255, 255), thickness=-1)
                            detection_FPGA_X = bodypart_coords_est_f[bp]["x"] - int(image_header["crop_x"])
                            detection_FPGA_Y = bodypart_coords_est_f[bp]["y"] - int(image_header["crop_y"])
                            detection_OPENCV_X = bodypart_coords_est_o[bp]["x"] - int(image_header["crop_x"])
                            detection_OPENCV_Y = bodypart_coords_est_o[bp]["y"] - int(image_header["crop_y"])
                            cv2.circle(displayFrame, (detection_FPGA_X, detection_FPGA_Y), 4, (0, 0, 255), thickness=-1) # FPGA - Red Color
                            cv2.circle(displayFrame, (detection_OPENCV_X, detection_OPENCV_Y), 4, (255, 255, 0), thickness=-1) # OPENCV - Cyan Color
                            cv2.circle(displayFrame, (bodypart_gt[frame_index]["bodypart_coords_gt"][detect_bodypart[bi]]["x"] - int(image_header["crop_x"]), bodypart_gt[frame_index]["bodypart_coords_gt"][detect_bodypart[bi]]["y"]- int(image_header["crop_y"])), 4, (255, 255, 255), thickness=-1)
                            cv2.imshow("voters", displayFrame)
                            cv2.waitKey(500)

            ack_message_f["detections"] = []
            for bi in allBodyParts:
                if (bi in bodypart_coords_est_f):
                    ack_message_f["detections"].append({"frame_index": frame_index,
                                                        "test_bodypart": bi,
                                                        "coord_x": bodypart_coords_est_f[bi]["x"],
                                                        "coord_y": bodypart_coords_est_f[bi]["y"],
                                                        "conf": bodypart_coords_est_f[bi]["conf"]} )
            ack_message_f = json.dumps(ack_message_f, separators=(',',':'))
            received_json_f = json.loads(ack_message_f)
            if ("detections" in received_json_f):
                for di in range(0, len(received_json_f["detections"])):
                    tbp = received_json_f["detections"][di]["test_bodypart"]
                    fi = received_json_f["detections"][di]["frame_index"]
                    if (tbp in bodypart_gt[fi]["bodypart_coords_gt"]):
                        error_stats = Error_Stats()
                        error_stats.frame_file = bodypart_gt[fi]["frame_file"]
                        error_stats.error_distance = np.sqrt(np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"] - received_json_f["detections"][di]["coord_x"]) + np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"] - received_json_f["detections"][di]["coord_y"]))
                        error_stats.conf = received_json_f["detections"][di]["conf"]
                        error_stats.annotation_x = bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"]
                        error_stats.annotation_y = bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"]
                        error_stats.detection_x = received_json_f["detections"][di]["coord_x"]
                        error_stats.detection_y = received_json_f["detections"][di]["coord_y"]

                        if (verbosity >= 1 ):
                            print "Frame Index: ",frame_index,"\nDistance between annotated and estimated", tbp, "location:", error_stats.error_distance
                        error_stats_all_f[tbp].append(error_stats)

            ack_message_o["detections"] = []
            for bi in allBodyParts:
                if (bi in bodypart_coords_est_o):
                    ack_message_o["detections"].append({"frame_index": frame_index,
                                                        "test_bodypart": bi,
                                                        "coord_x": bodypart_coords_est_o[bi]["x"],
                                                        "coord_y": bodypart_coords_est_o[bi]["y"],
                                                        "conf": bodypart_coords_est_o[bi]["conf"]} )
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
                        error_stats.conf = received_json_f["detections"][di]["conf"]
                        error_stats.annotation_x = bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"]
                        error_stats.annotation_y = bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"]
                        error_stats.detection_x = received_json_o["detections"][di]["coord_x"]
                        error_stats.detection_y = received_json_o["detections"][di]["coord_y"]

                        if (verbosity >= 1 ):
                            print "Frame Index:", frame_index, "\nDistance between annotated and estimated", tbp, "location:", error_stats.error_distance
                        error_stats_all_o[tbp].append(error_stats)

    print "..............Errors for FPGA Detection......."
    errorData_f = computeErrorStats(n_instance_gt, error_stats_all_f, testAnnotation_list)
    saveErrorStats(errorData_f, 'fpga')

    print ".............Errors for OpenCV Detection......."
    errorData_o = computeErrorStats(n_instance_gt, error_stats_all_o, testAnnotation_list)
    saveErrorStats(errorData_o, 'opencv')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--training-data-file", dest="training_data_file", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--knn-data-path", dest="knn_dir", default="", help="path containing knn data directory")
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

    testAnnotation_list = options.test_annotation_list
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
    knn_dir = options.knn_dir

    crop_size = options.crop_size

    train_list_pos_all = []
    train_list_neg_all = []

    testListAll = []
    with open(testAnnotation_list, 'r') as test_list:
        for tlist in test_list:
            testListAll.append(tlist)

    fN = options.training_data_file

    trainedDesc = []
    trainedRelPos = []
    with open(fN) as csvRead:
        for row in csv.reader(csvRead, delimiter="\t"):
            row = list(row)
            frow = []
            for item in row:
                frow.append(float(item))
            trainedDesc.append(list(frow[1:129]))
            trainedRelPos.append(list(frow[-2:]))

    detect_bodypart = train_bodypart
    bodypart = train_bodypart[0]

    detect(detect_bodypart, project_dir, knn_dir, testListAll, vote_sigma, vote_patch_size, vote_threshold, bodypart, trainedDesc, trainedRelPos)