#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import os
import re
import struct
import cv2
import numpy as np
import socket
import sys
import string
import copy

packer_header = struct.Struct('= 2s I')
packer_list_header = struct.Struct('= 2s I I')
packer_image_header = struct.Struct('= 2s I I I I')

class Error_Stats:
    def __init__(self):
        self.frame_file = None


def main(options, args):
    socks = []
    server_free = []
    n_server = 6
    for i in range(0, n_server):
        socks.append( socket.socket(socket.AF_INET, socket.SOCK_STREAM) )
        server_free.append(True)

    test_bodypart = options.test_bodypart

    test_annotations = []
    distance_coord_gt_interp_tot = []

    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.append(test_annotation)
                # test_FileNames.extend(test_annotation["VideoFile"])
                #test_annotations["VideoFile"].append(test_annotation["VideoFile"])
                #test_annotations["Annotations"].extend(test_annotation["Annotations"])
    print "len(test_annotations):", len(test_annotations)

    frame_index = -1
    error_stats_all = []

    annotatedPositionFile = string.join(['../expts/annotated_positions_',test_bodypart.strip(),".txt"])
    interpPositionFile = string.join(['../expts/interp_positions_',test_bodypart.strip(),".txt"])
    estimatedPositionFile = string.join(['../expts/estimated_positions_',test_bodypart.strip(),".txt"])

    distAnnotInterpFile = string.join(['../expts/distance_Annotated_&_Interp',test_bodypart.strip(),".txt"])
    distInterpEstimFile = string.join(['../expts/distance_Tnterp_&_Estim',test_bodypart.strip(),".txt"])
    distAnnotEstimFile = string.join(['../expts/distance_Annotated_&_Estim',test_bodypart.strip(),".txt"])

    meanError_withinInterpFile = string.join(['../expts/error_interpolation_',test_bodypart.strip(),".txt"])

    fileWriter_annotatedPosition = open(annotatedPositionFile,'w+')
    fileWriter_interpPosition = open(interpPositionFile,'w+')
    fileWriter_estimatedPosition = open(estimatedPositionFile,'w+')

    fileWriter_distAnnotInterp = open(distAnnotInterpFile,'w+')
    fileWriter_distInterpEstim = open(distInterpEstimFile,'w+')
    fileWriter_distAnnotEstim = open(distAnnotEstimFile,'w+')

    fileWriter_meanInterpolation = open(meanError_withinInterpFile,'w+')

    for i in range(0, len(test_annotations)):
        # bodypart_gt = {}

        bodypart_coords_gt = {}
        bodypart_coords_interp = {}
        bodypart_coords_est = {}
        distance_coord_gt_interp = {}
        distance_coord_est_interp = {}

        video_file = test_annotations[i]["VideoFile"]
        video_file = re.sub(".*/data/", "data/", video_file)
        video_file = os.path.join(options.project_dir, video_file)
        print "Video File: ", video_file
        cap = cv2.VideoCapture(video_file)

        bodypart_coords_gt[i]["VideoFile"] = video_file
        bodypart_coords_interp[i]["VideoFile"] = video_file
        bodypart_coords_est[i]["VideoFile"] = video_file

        if(cap.isOpened()):
            annotation = []
            annotation.extend(test_annotations[i]["Annotations"])
            n = 0
            for k in range(0, len(annotation)):
                cap.set(1, float(annotation[k]["FrameIndexVideo"]))
                # print "Number of frames", len(annotation)
                frame_index += 1
                n += 1
                ret, frame = cap.read()

                bodypart_coords_gt[i][k] = {}
                bodypart_coords_est[i][k] = {}
                bodypart_coords_interp[i][k] = {}

                if (options.display_level >= 2):
                    display_voters = frame.copy()

                for j in range(0, len(annotation[k]["FrameValueCoordinates"])):
                    if (annotation[k]["FrameValueCoordinates"][j]["Name"] == test_bodypart):
                        bodypart_coords_gt[i][k]["x"] = int(annotation[k]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                        bodypart_coords_gt[i][k]["y"] = int(annotation[k]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])
                        bodypart_coords_gt[i][k]["FrameIndexVideo"] = int(annotation[k]["FrameIndexVideo"])
                        bodypart_coords_gt[i][k]["BodyPart"] = test_bodypart

                if ( bodypart_coords_gt[i][k] == None ):
                    print "Empty Image Video %d, Frame %d" %(i,k)
                    continue

                print "frame_index:", frame_index

                bodypart_coords_est[i][k]["x"] = bodypart_coords_gt[i][k]["x"]
                bodypart_coords_est[i][k]["y"] = bodypart_coords_gt[i][k]["x"]
                bodypart_coords_est[i][k]["FrameIndexVideo"] = bodypart_coords_gt[i][k]["FrameIndexVideo"]
                bodypart_coords_est[i][k]["BodyPart"] = bodypart_coords_gt[i][k]["Bodypart"]

                bodypart_coords_interp[i][k]["x"] = bodypart_coords_gt[i][k]["x"]
                bodypart_coords_interp[i][k]["y"] = bodypart_coords_gt[i][k]["x"]
                bodypart_coords_interp[i][k]["FrameIndexVideo"] = bodypart_coords_gt[i][k]["FrameIndexVideo"]
                bodypart_coords_interp[i][k]["BodyPart"] = bodypart_coords_gt[i][k]["Bodypart"]

                # perform detection
                images = []
                packed_image_headers = []
                images_data = []

                image = copy.deepcopy(frame)

                print "bodypart_coords_gt:" , bodypart_coords_gt[i][k]
                # bodypart_coords_interp[k] = bodypart_coords_gt[k]
                # print "bodypart_coords_interp:" , bodypart_coords_interp[k]

                crop_x = 0
                crop_y = 0
                # image = image[crop_y:1920,crop_x:1920,0]

                images.append(copy.deepcopy(image))

                image_info = np.shape(image)
                if not image_info:
                    print "Empty Image Video %d, Frame %d" %(i,k)
                    continue
                print image_info
                image_header = ('01', image_info[0], image_info[1], crop_x, crop_y)
                packed_image_header = packer_image_header.pack(*image_header)
                packed_image_headers.append(copy.deepcopy(packed_image_header))

                image_data = np.getbuffer(np.ascontiguousarray(image))
                print len(image_data)
                images_data.append(image_data)

                list_header = ('00', frame_index, len(images_data))
                packed_list_header = packer_list_header.pack(*list_header)

                blob_size = len(packed_list_header)
                for j in range(0, len(images_data)):
                    blob_size += len(packed_image_headers[j]) + len(images_data[j])

                header = ('01', blob_size)
                packed_header = packer_header.pack(*header)

                received_json = None

                for s in range(0, n_server):
                    if ( server_free[s] ):
                        try:
                            print "trying to send packet to server", s

                            HOST, PORT = "localhost", 9988 + s
                            if ( socks[s] == None ):
                                socks[s] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                            socks[s].connect((HOST, PORT))
                            socks[s].setblocking(1)

                            socks[s].sendall(packed_header)

                            socks[s].sendall(packed_list_header)

                            for j in range(0, len(images_data)):
                                socks[s].sendall(packed_image_headers[j])
                                socks[s].sendall(images_data[j])

                            print "Sent packet on server", s
                            server_free[s] = False
                        finally:
                            break

                for s in range(0, n_server):
                    if ( not server_free[s] ):
                        # Receive data from the server and shut down
                        try:
                            socks[s].settimeout(1/200.0)
                            received = socks[s].recv(1024)
                            socks[s].close()
                            socks[s] = None
                            server_free[s] = True
                            print "Received from server {} : {}".format(s, received)
                            received_json = json.loads(received)

                            if ("detections" in received_json):
                                for di in range(0, len(received_json["detections"])):
                                    if ( received_json["detections"][di]["test_bodypart"] == test_bodypart ):
                                        bodypart_coords_est[i][k]["x"] = received_json["detections"][di]["coord_x"]
                                        bodypart_coords_est[i][k]["y"] = received_json["detections"][di]["coord_y"]
                                        f_index = received_json["detections"][di]["frame_index"]
                                    else:
                                        bodypart_coords_est[i][k]["x"] = -1
                                        bodypart_coords_est[i][k]["y"] = -1
                        except:
                            pass

                        error_stats = Error_Stats()
                        error_stats.video_file = bodypart_coords_gt[i]["VideoFile"]
                        error_stats.frame_number = bodypart_gt[i]["FrameIndexVideo"]
                        error_stats.error_distance = np.sqrt(np.square(bodypart_coords_gt[i][k]["x"] - bodypart_coords_est[i][k]["x"]) +
                                                                 np.square(bodypart_coords_gt[i][k]["y"] - bodypart_coords_est[i][k]["y"]))

                        # if (bodypart_coords_est is not None):
                        #     print "Calculating Errors......."
                        #     cv2.waitKey(100)
                        #
                        #     error_stats.error_distance_interpolation_est = np.sqrt(np.square(bodypart_gt[f_index]["bodypart_coords_interp"]["x"] - bodypart_coords_est["x"]) +
                        #                                                            np.square(bodypart_gt[f_index]["bodypart_coords_interp"]["y"] - bodypart_coords_est["y"]))
                        #     error_stats.error_distance_interpolation_annot = np.sqrt(np.square(bodypart_gt[f_index]["bodypart_coords_interp"]["x"] - bodypart_coords_gt["x"]) +
                        #                                                              np.square(bodypart_gt[f_index]["bodypart_coords_interp"]["y"] - bodypart_coords_gt["y"]))
                        #     print bodypart_gt[f_index]["video_file"],bodypart_gt[f_index]["frame_number"], "Distance between annotated and estimated RightMHhook location:", error_stats.error_distance
                        #     print bodypart_gt[f_index]["video_file"],bodypart_gt[f_index]["frame_number"], "Distance between interp and annotated RightMHhook location:", error_stats.error_distance_interpolation_annot
                        #     print bodypart_gt[f_index]["video_file"],bodypart_gt[f_index]["frame_number"], "Distance between interp and estimated RightMHhook location:", error_stats.error_distance_interpolation_est
                        #     error_stats_all.append(error_stats)

                print server_free
                fileWriter_2.writelines(str(bodypart_coords_gt[k]) + "\n")
                fileWriter_3.writelines(str(bodypart_coords_interpolated[k]) + "\n")
        else:
            print "Not able to read video file"

        # fileWriter_annotatedPosition.writelines(str(bodypart_coords_gt[i][k]) + "\n")
        json.dump(bodypart_coords_gt[i],fileWriter_annotatedPosition)
        fileWriter_estimatedPosition.writelines(str(bodypart_coords_est[i][k]) + "\n")
        json.dump(bodypart_coords_est[i],fileWriter_estimatedPosition)

        distance_coord_gt_interp[i] =[]
        flag[i] = np.zeros(shape=(len(bodypart_coords_gt),1))
        k = 0
        for m in range(0, len(bodypart_coords_gt)):
            if (m > 0 and m < ((len(bodypart_coords_gt)-1))):
                if ((bodypart_coords_gt[m-1]["x"] != -1) and (bodypart_coords_gt[m+1]["x"] != -1) and (bodypart_coords_gt[m]["x"] != -1)):
                    flag[i][m] = 1
                    k += 1
                    bodypart_coords_interp[m]["x"] = int((bodypart_coords_gt[m-1]["x"]+bodypart_coords_gt[m+1]["x"])/2)
                    bodypart_coords_interp[m]["y"] = int((bodypart_coords_gt[m-1]["y"]+bodypart_coords_gt[m+1]["y"])/2)
                    tempDist = np.sqrt(np.square(bodypart_coords_gt[m]["x"]-bodypart_coords_interp[m]["x"]) + np.square(bodypart_coords_gt[m]["y"]- bodypart_coords_interp[m]["y"]))
                    distance_coord_gt_interp.append(tempDist)
                    distance_coord_gt_interp_tot.append(tempDist)
                else:
                    flag[i][m] = 0
                    bodypart_coords_interp[m]["x"] = int(bodypart_coords_gt[m]["x"])
                    bodypart_coords_interp[m]["y"] = int(bodypart_coords_gt[m]["y"])
            else:
                if bodypart_coords_gt[m]["x"] != -1
                    flag[i][m] = 1
                bodypart_coords_interp[m]["x"] = int(bodypart_coords_gt[m]["x"])
                bodypart_coords_interp[m]["y"] = int(bodypart_coords_gt[m]["y"])

<<<<<<< HEAD
            # distance_coord_est_interp[i] = np.sqrt(np.square(bodypart_coords_est[i]["x"]-bodypart_coords_interpolated[i]["x"]) + np.square(bodypart_coords_est[i]["y"]- bodypart_coords_interpolated[i]["y"]))
            # distance_coord_est_interp[i] = np.sqrt(np.square(bodypart_coords_est[i]["x"]-bodypart_coords_interp[i]["x"]) + np.square(bodypart_coords_est[i]["y"]- bodypart_coords_interp[i]["y"]))
=======
            # distance_coord_est_interp[i] = np.sqrt(np.square(bodypart_coords_est[i]["x"]-bodypart_coords_interp[i]["x"]) + np.square(bodypart_coords_est[i]["y"]- bodypart_coords_interp[i]["y"]))

            # distance_coord_est_interp[i] = np.sqrt(np.square(bodypart_coords_est[i]["x"]-bodypart_coords_interpolated[i]["x"]) + np.square(bodypart_coords_est[i]["y"]- bodypart_coords_interpolated[i]["y"]))

>>>>>>> ad
        # json.dump(distance_coord_gt_interp, fileWriter)
        # strPrint = string.join(['Distance between interp and annotated position for file : ', i,'\t', distance_coord_gt_interp,'\n'])
        # fileWriter.writelines(str(bodypart_coords_gt) + "\n")

<<<<<<< HEAD
        # fileWriter.writelines(str(bodypart_coords_interpolated) + "\n")
        fileWriter_1.writelines(str(np.mean(distance_coord_gt_interp[k])) + "\n")
        fileWriter_4.writelines(str(distance_coord_gt_interp[k]) + "\n")

=======
>>>>>>> ad
        # fileWriter.writelines(str(bodypart_coords_interp) + "\n")
        fileWriter_annotatedPosition.writelines(str(bodypart_coords_gt[k]) + "\n")
        fileWriter_interpPosition.writelines(str(bodypart_coords_gt[k]) + "\n")
        fileWriter_annotatedPosition.writelines(str(distance_coord_gt_interp[k]) + "\n")
        fileWriter_meanInterpolation.writelines(str(np.mean(distance_coord_gt_interp[k])) + "\n")
<<<<<<< HEAD

=======
        # fileWriter.writelines(str(bodypart_coords_interpolated) + "\n")
        fileWriter_1.writelines(str(np.mean(distance_coord_gt_interp[k])) + "\n")
        fileWriter_4.writelines(str(distance_coord_gt_interp[k]) + "\n")
>>>>>>> ad
    fileWriter_4.writelines(str(distance_coord_gt_interp_tot[k]) + "\n")
    fileWriter_4.writelines(string.join(["Total Mean for ",str(len(distance_coord_gt_interp_tot))," number of frames :", str(np.mean(distance_coord_gt_interp_tot)),"\n"]))
    # print distance_coord_est_interp

    # print os.sys.argv
    # error_distance_inliers = []
    # for es in error_stats_all:
    #     if (es.error_distance <= options.outlier_error_dist):
    #         error_distance_inliers.append(es.error_distance)
    # n_outlier = len(error_stats_all) - len(error_distance_inliers)
    # print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (options.outlier_error_dist, n_outlier, len(error_stats_all), float(n_outlier) /float(max(1, float(len(error_stats_all)))))
    # print "Median inlier error dist =", np.median(error_distance_inliers)
    # print "Mean inlier error dist =", np.mean(error_distance_inliers)
    # print "len(bodypart_gt):", len(bodypart_gt)

    cv2.destroyWindow("frame")
    fileWriter_1.close()
    fileWriter_2.close()
    fileWriter_3.close()
    fileWriter_4.close()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="", help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--test-bodypart", dest="test_bodypart",default="MouthHook", help="Input the bodypart to be tested")

    (options, args) = parser.parse_args()

    main(options, args)