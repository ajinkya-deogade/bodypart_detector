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

global allBodyParts
allBodyParts = ['MouthHook', 'LeftMHhook', 'RightMHhook', 'LeftDorsalOrgan', 'RightDorsalOrgan']

class KeyPoint:
   def __init__(self, frame_id, x, y, angle, rel_x, rel_y, bp):
        self.frame_id = frame_id
        self.pt = (x, y)
        self.angle = angle
        self.rel_pt = (rel_x, rel_y)
        self.bodypart = bp

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
        frameNumber = unpack_from(fmt_spec+'1'+int_spec, frm)
        frameNumber = frameNumber[0]
        while frameNumber > 0:
            numberKP = unpack_from(fmt_spec+int_spec, f.read(intBytes))
            numberKP = numberKP[0]
            data['frameNumber'].append(frameNumber+1)
            data['position'][0].append(list(unpack_from(fmt_spec + str(numberKP) + double_spec, f.read(numberKP*doubleBytes))))
            data['position'][1].append(list(unpack_from(fmt_spec + str(numberKP) + double_spec, f.read(numberKP*doubleBytes))))
            data['angle'].append(list(unpack_from(fmt_spec + str(numberKP) + double_spec, f.read(numberKP*doubleBytes))))
            data['relative_position'][0].append(list(unpack_from(fmt_spec + str(numberKP) + double_spec, f.read(numberKP*doubleBytes))))
            data['relative_position'][1].append(list(unpack_from(fmt_spec + str(numberKP) + double_spec, f.read(numberKP*doubleBytes))))
            frameNumber = -1
            frm = f.read(intBytes)
            if not frm:
                break
            frameNumber = unpack_from(fmt_spec+'1'+int_spec, frm)
            frameNumber = frameNumber[0]
    kpData = []
    for i in range(0, len(data['frameNumber'])):
        kpData_frame = []
        for j in range(0, len(data['position'][0][i])):
            kp2 = KeyPoint(data['frameNumber'][i],data['position'][0][i][j],data['position'][1][i][j],data['angle'][i][j],data['relative_position'][0][i][j],data['relative_position'][1][i][j], bodyPart)
            kpData_frame.append(kp2)
        kpData.append(kpData_frame)

    return kpData

def computeVoteMapFPGA(bodypart_vote_map, bodypart_vote, keypointData, frame, vote_patch_size):
    for knn_id in range(0, len(keypointData)):
        relative_distance = 0
        if (relative_distance <= desc_distance_threshold):
            kp = keypointData[knn_id]
            a = (kp.angle*3.14)
            if a<0:
                a += 6.28
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

            for bi in range(0, len(allBodyParts)):
                if allBodyParts[bi] == kp.bodypart:
                    vote_loc = kp.rel_pt
                    vote_bodypart = kp.bodypart
                    vi = -1
                    for vid in range(0, len(detect_bodypart)):
                        if ( detect_bodypart[vid] == vote_bodypart):
                            vi = vid
                            break
                    if( vi == -1 ):
                        continue
                    kpy, kpx = kp.pt
                    # p = (kpy, kpx) + np.dot(R, vote_loc)
                    # p = (kpy, kpx) + np.dot(R, vote_loc)
                    dotP = np.dot(R, vote_loc)
                    x = kpy + dotP[0]
                    y = kpx + dotP[1]
                    if not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size):
                        y_start = int(float(y)) - int(float(vote_patch_size))
                        y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                        x_start = int(float(x)) - int(float(vote_patch_size))
                        x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                        bodypart_vote_map[bi][y_start:y_end, x_start:x_end] += bodypart_vote

    return bodypart_vote_map

def computeVoteMapOPENCV(bodypart_vote_map, bodypart_vote, bodypart_knn_pos, surf, trainedRelPos, frame, vote_patch_size):
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

                    for bi in range(0, len(trainedRelPos[r_pos])):
                        vote_bodypart = "MouthHook"
                        vote_loc = trainedRelPos[r_pos]
                        vi = -1
                        for vid in range(0, len(detect_bodypart)):
                            if ( detect_bodypart[vid] == vote_bodypart):
                                vi = vid
                                break
                        if( vi == -1 ):
                            continue
                        p = kp_frame[h].pt + np.dot(R, vote_loc)
                        x, y = p
                        if not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size):
                            y_start = int(float(y)) - int(float(vote_patch_size))
                            y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                            x_start = int(float(x)) - int(float(vote_patch_size))
                            x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                            bodypart_vote_map[vi][y_start:y_end, x_start:x_end] += bodypart_vote
    return bodypart_vote_map

def detect(detect_bodypart, project_dir, train_data_positive, train_data_negative, test_annotation_list, vote_sigma, vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, kpData, bodypart, trainedDesc, trainedRelPos):
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

    error_stats_all = {}
    n_instance_gt = {}
    n_instance_gt["MouthHook"] = 0
    for s in detect_bodypart:
        error_stats_all[s] = []
        n_instance_gt[s] = 0

    bodypart_gt = {}
    verbosity = 0
    display_level = 3

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

        print "Video Frame: ", frameIndexVideo
        keypointData = kpData[int(frameIndexVideo)]

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

        surf = cv2.xfeatures2d.SURF_create(250, nOctaves=2, nOctaveLayers=3, extended=1)

        bodypart_knn_pos = FLANN()
        bodypart_knn_pos.build_index(np.array(trainedDesc, dtype=np.float32), params=dict(algorithm=1, trees=4))

        if (verbosity >= 2):
            print "bodypart_coords_gt:", bodypart_coords_gt

        if "MouthHook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
            crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-int(crop_size/2))
            crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-int(crop_size/2))
            frame = frameOrg[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size,0]

            image_info = np.shape(frame)
            if (verbosity >= 2):
                print image_info

            bodypart_vote_map_fp = []
            for bid in range(0, len(detect_bodypart)):
                bodypart_vote_map_fp.append(np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float))

            bodypart_vote_map_op = []
            for bid in range(0, len(detect_bodypart)):
                bodypart_vote_map_op.append(np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float))


            bodypart_vote = np.zeros((2 * vote_patch_size + 1, 2 * vote_patch_size + 1), np.float)
            for x in range(-vote_patch_size, vote_patch_size + 1):
                for y in range(-vote_patch_size, vote_patch_size + 1):
                    bodypart_vote[y + vote_patch_size, x + vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(vote_sigma))) / (vote_sigma * np.sqrt(2 * np.pi))

            if (verbosity >= 2):
                print "Bodypart Vote: ", np.shape(bodypart_vote)

            ack_message={}
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
            voteMapOPENCV = computeVoteMapOPENCV(bodypart_vote_map_op, bodypart_vote, bodypart_knn_pos, surf, trainedRelPos, frame, vote_patch_size)
        else:
            continue

        for bi in range(0, len(detect_bodypart)):
            vote_max_f = np.amax(voteMapFPGA[bi][:,:])
            vote_max_o = np.amax(voteMapOPENCV[bi][:,:])
            if (vote_max_f > vote_threshold and ((detect_bodypart[bi] not in bodypart_coords_est_f) or vote_max_f > bodypart_coords_est_f[detect_bodypart[bi]]["conf"])) and (vote_max_o > vote_threshold and ((detect_bodypart[bi] not in bodypart_coords_est_o) or vote_max_o > bodypart_coords_est_o[detect_bodypart[bi]]["conf"])):
                vote_max_loc_f = np.array(np.where(voteMapFPGA[bi][:,:] == vote_max_f))
                vote_max_loc_o = np.array(np.where(voteMapOPENCV[bi][:,:] == vote_max_o))
                vote_max_loc_f = vote_max_loc_f[:,0]
                vote_max_loc_o = vote_max_loc_o[:,0]
                bodypart_coords_est_f[detect_bodypart[bi]] = {"conf" : vote_max_f, "x" : int(vote_max_loc_f[1]) + int(image_header["crop_x"]), "y" : int(vote_max_loc_f[0]) + int(image_header["crop_y"])}
                bodypart_coords_est_o[detect_bodypart[bi]] = {"conf" : vote_max_o, "x" : int(vote_max_loc_o[1]) + int(image_header["crop_x"]), "y" : int(vote_max_loc_o[0]) + int(image_header["crop_y"])}
                displayFrame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).copy()
                # displayFrame = frameOrg.copy()
                # rotMat = cv2.getRotationMatrix2D((128,128), 90, 1)
                # rows,cols = displayFrame.shape
                # displayFrame_2 = cv2.warpAffine(displayFrame, rotMat, displayFrame.size)
                # displayFrame_2 = np.transpose(displayFrame[:, :, 1])
                if (display_level >= 2):
                    if ("x" in bodypart_coords_est_f[detect_bodypart[bi]] and "x" in bodypart_coords_est_o[detect_bodypart[bi]]):
                        # print "FPGA: ", bodypart_coords_est_f[detect_bodypart[bi]]
                        # print "OPENCV: ", bodypart_coords_est_o[detect_bodypart[bi]]
                        # print "Annotated: ", bodypart_gt[frame_index]["bodypart_coords_gt"][detect_bodypart[bi]]
                        # print "Image Head", image_header
                        for kk in keypointData:
                            # cv2.circle(displayFrame, (int(kk.pt[0])+int(image_header["crop_x"]), int(kk.pt[1])+int(image_header["crop_y"])), 4, (0, 255, 255), thickness=-1)
                            cv2.circle(displayFrame, (int(kk.pt[0]), int(kk.pt[1])), 4, (0, 255, 255), thickness=-1)
                        cv2.circle(displayFrame, (bodypart_coords_est_f[detect_bodypart[bi]]["x"] - int(image_header["crop_x"]), bodypart_coords_est_f[detect_bodypart[bi]]["y"]- int(image_header["crop_y"])), 4, (255, 0, 0), thickness=-1)
                        # cv2.circle(displayFrame, (bodypart_coords_est_f[detect_bodypart[bi]]["x"], bodypart_coords_est_f[detect_bodypart[bi]]["y"]), 4, (255, 0, 0), thickness=-1)
                        # cv2.circle(displayFrame, (bodypart_coords_est_o[detect_bodypart[bi]]["x"] - int(image_header["crop_x"]), bodypart_coords_est_o[detect_bodypart[bi]]["y"]- int(image_header["crop_y"])), 4, (0, 0, 255), thickness=-1)
                        # cv2.circle(displayFrame, (bodypart_gt[frame_index]["bodypart_coords_gt"][detect_bodypart[bi]]["x"] - int(image_header["crop_x"]), bodypart_gt[frame_index]["bodypart_coords_gt"][detect_bodypart[bi]]["y"]- int(image_header["crop_y"])), 4, (0, 255, 0), thickness=-1)
                    cv2.imshow("voters", displayFrame)
                    cv2.waitKey(500)

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
    # global allBodyParts

    train_data_positive = '../expts/20160218_185439_fragmented/positive.p'
    train_data_negative = '../expts/20160218_185439_fragmented/negative.p'
    fpgaKNN = '../expts/NN_KP_20160308_185227.bin'

    testListAll = []
    test_annotation_list = '../config/train_annotation_list'
    with open(test_annotation_list, 'r') as testList:
        for tlist in testList:
            testListAll.append(tlist)

    fN = '/Users/Ajinkya/work/dorsalOrganDetection/mhdo/experiments/ad_69/expts/TrainedDataPosMHSortedEdited.txt'
    trainedDesc = []
    trainedRelPos = []
    f = open(fN)
    ind = -1
    for row in csv.reader(f, delimiter="\t"):
        ind += 1
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
    bodypart = "MouthHook"
    fpga_kp = readFPGAKP(fpgaKNN, bodypart)
    detect(detect_bodypart, project_dir, train_data_positive, train_data_negative, testListAll, vote_sigma, vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, fpga_kp, bodypart, trainedDesc, trainedRelPos)
