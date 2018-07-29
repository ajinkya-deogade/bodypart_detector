#! /opt/local/bin/python

import numpy as np
import cv2
from struct import *
import json
from pyflann import *
import re
from optparse import OptionParser
import time

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
            kp = KeyPoint(data['frameNumber'][i],data['position'][0][i][j],data['position'][1][i][j],data['angle'][i][j],data['relative_position'][0][i][j],data['relative_position'][1][i][j], bodyPart)
            kpData_frame.append(kp)
        kpData.append(kpData_frame)

    return kpData

def detect(detect_bodypart, project_dir, train_data_positive, train_data_negative, test_annotation_list, vote_sigma, vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, kpData, bodypart):
    # print "Performing detections......."
    # test_annotations = []
    #
    # for test_annotation_file in test_annotation_list_fpgaKNNVal:
    #     print test_annotation_file
    #     test_annotation_file = os.path.join(project_dir, re.sub(".*/data/", "data/", test_annotation_file.strip()))
    #     with open(test_annotation_file) as fin_annotation:
    #         test_annotation = json.load(fin_annotation)
    #         test_annotations.extend(test_annotation["Annotations"])
    # print "len(test_annotations):" , len(test_annotations)

    frame_index = -1

    # error_stats_all = {}
    # n_instance_gt = {}
    # n_instance_gt["MouthHook"] = 0
    # for s in detect_bodypart:
    #     error_stats_all[s] = []
    #     n_instance_gt[s] = 0
    #
    # bodypart_gt = {}
    # verbosity = 0
    # display_level = 3
    video_file = '/Users/Ajinkya/work/dorsalOrganDetection/data/Janelia_Q2_2015/20150501_MPEG4_NoOdor/Clips/001_20150430_171054_s60_clip_001.mp4'
    print "Video File: ", video_file
    cap = cv2.VideoCapture(video_file)

    while(True):
        ret, frameOrg = cap.read()
        frame = cv2.cvtColor(frameOrg, cv2.COLOR_BGR2GRAY)

    for j in range(0, len(test_annotations)):
        frame_index += 1
        # os.system('clear')
        print "Percentage Complete: %.2f" %(float(frame_index)/float(len(test_annotations))*100)

        annotation = test_annotations[j]

        frame_file = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(project_dir, frame_file)
        frame = cv2.imread(frame_file)
        frameIndexVideo = annotation["FrameIndexVideo"]

        if (display_level >= 2):
            display_voters = frame.copy()

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

        if (verbosity >= 2):
            print "bodypart_coords_gt:" , bodypart_coords_gt

        if "MouthHook" in bodypart_gt[frame_index]["bodypart_coords_gt"]:
            crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-int(crop_size/2))
            crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-int(crop_size/2))
            frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size,0]

            image_info = np.shape(frame)
            if (verbosity >= 2):
                print image_info

            bodypart_vote_map = []
            for bid in range(0, len(detect_bodypart)):
                bodypart_vote_map.append(np.zeros((np.shape(frame)[0], np.shape(frame)[1]), np.float))

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

            if (display_level >= 2):
                display_voters = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            for knn_id in range(0, len(keypointData)):
                # r_pos = r_pos_all[:,knn_id]
                # d_pos = d_pos_all[:,knn_id]
                # relative_distance = d_pos - d_neg
                relative_distance = 0

                if (relative_distance <= desc_distance_threshold):
                    # a = np.pi * data['angle'][knn_id]/ 180.0
                    kp = keypointData[knn_id]
                    a = (kp.angle*3.14)
                    if a<0:
                        a += 6.28
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

                    for bi in range(0, len(allBodyParts)):
                        # print kp.bodypart
                        # print allBodyParts[bi]
                        # raw_input("Enter...")
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
                            p = kp.pt + np.dot(R, vote_loc)
                            x, y = p
                            if (not (x <= vote_patch_size or x >= np.shape(frame)[1] - vote_patch_size or  y <= vote_patch_size or y >= np.shape(frame)[0] - vote_patch_size)):
                                y_start = int(float(y)) - int(float(vote_patch_size))
                                y_end = int(float(y)) + int(float(vote_patch_size) + 1.0)
                                x_start = int(float(x)) - int(float(vote_patch_size))
                                x_end = int(float(x)) + int(float(vote_patch_size) + 1.0)
                                bodypart_vote_map[vi][y_start:y_end, x_start:x_end] += bodypart_vote
        else:
            continue

        if (display_level >= 3):
            cv2.circle(display_voters, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)
            cv2.waitKey(1000)

        for bi in range(0, len(detect_bodypart)):
            vote_max = np.amax(bodypart_vote_map[bi][:,:])
            if (vote_max > vote_threshold and ((detect_bodypart[bi] not in bodypart_coords_est) or vote_max > bodypart_coords_est[detect_bodypart[bi]]["conf"])):
                vote_max_loc = np.array(np.where(bodypart_vote_map[bi][:,:] == vote_max))
                vote_max_loc = vote_max_loc[:,0]
                bodypart_coords_est[detect_bodypart[bi]] = {"conf" : vote_max, "x" : int(vote_max_loc[1]) + int(image_header["crop_x"]), "y" : int(vote_max_loc[0]) + int(image_header["crop_y"])}

                if (display_level >= 2):
                    display_vote_map = np.array(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).copy(), np.float)
                    display_vote_map /= 255.0
                    bodypart_vote_map /= np.amax(bodypart_vote_map[bi][:, :])
                    # print np.shape(bodypart_vote_map[bi])
                    # raw_input('Enter')
                    display_vote_map[:, :, 2] = bodypart_vote_map[bi]
                    if ("x" in bodypart_coords_est[detect_bodypart[bi]]):
                        cv2.circle(display_vote_map, (bodypart_coords_est[detect_bodypart[bi]]["x"] - int(image_header["crop_x"]), bodypart_coords_est[detect_bodypart[bi]]["y"]- int(image_header["crop_y"])), 4, (0, 255, 255), thickness=-1)
                        cv2.circle(display_vote_map, (bodypart_gt[frame_index]["bodypart_coords_gt"][detect_bodypart[bi]]["x"] - int(image_header["crop_x"]), bodypart_gt[frame_index]["bodypart_coords_gt"][detect_bodypart[bi]]["y"]- int(image_header["crop_y"])), 4, (0, 255, 0), thickness=-1)
                    # display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("voters", display_vote_map)
                    cv2.waitKey(500)

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

    testList = '../expts/' + timestr + '_testList.lst'
    listWriter = open(testList, 'w')
    for testFile in test_annotation_list:
        listWriter.write(testFile)
    listWriter.close()

    resultFileName = '../expts/' + timestr + '_results.log'
    res = open(resultFileName, 'w')
    testData = {}

    testData["DetectionParameters"] = {}
    testData["DetectionParameters"]["PositiveTrainFile"] = train_data_positive
    testData["DetectionParameters"]["NegativeTrainFile"] = train_data_negative
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

    json.dump(testData, res)
    res.close()

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

    detect_bodypart = train_bodypart
    bodypart = "MouthHook"
    fpga_kp = readFPGAKP(fpgaKNN, bodypart)
    detect(detect_bodypart, project_dir, train_data_positive, train_data_negative, testListAll, vote_sigma, vote_patch_size, desc_distance_threshold, vote_threshold, outlier_error_dist, fpga_kp, bodypart)