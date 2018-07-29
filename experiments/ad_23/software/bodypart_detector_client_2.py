#! /usr/bin/env python

from optparse import OptionParser
import json
import os
import re
import struct
import cv2
import numpy as np
import socket
import sys
import copy
import time

packer_header = struct.Struct('= 2s I')
packer_list_header = struct.Struct('= 2s I I')
packer_image_header = struct.Struct('= 2s I I I I')
unpacker_ack_header = struct.Struct('= 2s I')


class Error_Stats:
    def __init__(self):
        self.frame_file = None


def main(options, args):
    socks = []
    server_free = []
    n_server = 4
    confidence_votes = []
    np.seterr(divide='ignore', invalid='ignore')

    for videoNumber in range(0, n_server):
        socks.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        server_free.append(True)

    test_bodypart = options.test_bodypart
    total_time_frame = []

    test_annotations = []
    tempDist_vector_gt_interp_tot = []
    tempDist_vector_gt_estim_tot = []
    tempDist_vector_estim_interp_tot = []
    confidence_votes_1 = []
    confidence_votes_2 = []
    confidence_votes_3 = []
    confidence_votes_4 = []
    confidence_votes_5 = []
    confidence_votes_6 = []
    confidence_votes_7 = []
    confidence_votes_8 = []
    confidence_votes_None = []
    confidence_votes_distribution = {}
    timeStats = {}

    bodypart_coords_gt = {}
    bodypart_coords_interp = {}
    bodypart_coords_est = {}

    distance_coord_gt_interp = {}
    distance_coord_gt_estim = {}
    distance_coord_estim_interp = {}

    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,
                                                re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.append(test_annotation)
                # test_FileNames.extend(test_annotation["VideoFile"])
                # test_annotations["VideoFile"].append(test_annotation["VideoFile"])
                # test_annotations["Annotations"].extend(test_annotation["Annotations"])
    print "len(test_annotations):", len(test_annotations)

    frame_index = -1
    save_folder = options.save_folder
    annotatedPositionFile = os.path.join(save_folder+'annotated_positions_' + test_bodypart.strip() + ".json")
    interpPositionFile = os.path.join(save_folder+'interp_positions_' + test_bodypart.strip() + ".json")
    estimatedPositionFile = os.path.join(save_folder+'estimated_positions_' + test_bodypart.strip() + ".json")

    distAnnotInterpFile = os.path.join(save_folder+'distance_Annotated_&_Interp_' + test_bodypart.strip() + ".json")
    distInterpEstimFile = os.path.join(save_folder+'distance_Interp_&_Estim_' + test_bodypart.strip() + ".json")
    distAnnotEstimFile = os.path.join(save_folder+'distance_Annotated_&_Estim_' + test_bodypart.strip() + ".json")
    allMeansMediansFile = os.path.join(save_folder+'allMeans&Medians_' + test_bodypart.strip() + ".json")
    allTimeFile = os.path.join(save_folder+'allTime_' + test_bodypart.strip() + ".json")
    confidenceFile = os.path.join(save_folder+'confidence_' + test_bodypart.strip() + ".json")

    fileWriter_annotatedPosition = open(annotatedPositionFile, 'w+')
    fileWriter_interpolatedPosition = open(interpPositionFile, 'w+')
    fileWriter_estimatedPosition = open(estimatedPositionFile, 'w+')

    fileWriter_distAnnotInterp = open(distAnnotInterpFile, 'w+')
    fileWriter_distInterpEstim = open(distInterpEstimFile, 'w+')
    fileWriter_distAnnotEstim = open(distAnnotEstimFile, 'w+')
    fileWriter_allMeans = open(allMeansMediansFile, 'w+')
    fileWriter_allTime = open(allTimeFile, 'w+')
    fileWriter_confidence = open(confidenceFile, 'w+')

    for videoNumber in range(0, len(test_annotations)):

        bodypart_coords_gt[videoNumber] = {}
        bodypart_coords_interp[videoNumber] = {}
        bodypart_coords_est[videoNumber] = {}

        distance_coord_gt_interp[videoNumber] = {}
        distance_coord_gt_estim[videoNumber] = {}
        distance_coord_estim_interp[videoNumber] = {}

        video_file = test_annotations[videoNumber]["VideoFile"]
        video_file = re.sub(".*/data/", "data/", video_file)
        video_file = os.path.join(options.project_dir, video_file)
        print "Video File: ", video_file
        cap = cv2.VideoCapture(video_file)

        bodypart_coords_gt[videoNumber]["VideoFile"] = video_file
        bodypart_coords_interp[videoNumber]["VideoFile"] = video_file
        bodypart_coords_est[videoNumber]["VideoFile"] = video_file

        distance_coord_gt_interp[videoNumber]["VideoFile"] = video_file
        distance_coord_gt_estim[videoNumber]["VideoFile"] = video_file
        distance_coord_estim_interp[videoNumber]["VideoFile"] = video_file

        bodypart_coords_gt[videoNumber]["StartFrames"] = []
        bodypart_coords_gt[videoNumber]["EndFrames"] = []

        if (cap.isOpened()):
            annotation = []
            annotation.extend(test_annotations[videoNumber]["Annotations"])
            start_f = None
            end_f = None
            for frameNumber in range(0, len(annotation)):
                for j in range(0, len(annotation[frameNumber]["FrameValueCoordinates"])):
                    if (annotation[frameNumber]["FrameValueCoordinates"][j]["Name"] == test_bodypart):
                        if (annotation[frameNumber]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                            if frameNumber > 0 and frameNumber < len(annotation)-1:
                                if (annotation[frameNumber-1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] == -1)and (annotation[frameNumber+1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                                        start_f = annotation[frameNumber]["FrameIndexVideo"]
                                        bodypart_coords_gt[videoNumber]["StartFrames"].append(start_f)
                                if ((annotation[frameNumber-1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1)and (annotation[frameNumber+1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] == -1)):
                                        end_f = annotation[frameNumber]["FrameIndexVideo"]
                                        bodypart_coords_gt[videoNumber]["EndFrames"].append(end_f)
                            elif frameNumber == 0:
                                if (annotation[frameNumber+1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                                    start_f = annotation[frameNumber]["FrameIndexVideo"]
                                    bodypart_coords_gt[videoNumber]["StartFrames"].append(start_f)
                            elif frameNumber == len(annotation)-1:
                                if (annotation[frameNumber-1]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1):
                                        end_f = annotation[frameNumber]["FrameIndexVideo"]
                                        bodypart_coords_gt[videoNumber]["EndFrames"].append(end_f)

            for frameNumber in range(0, len(annotation)):
                cap.set(1, float(annotation[frameNumber]["FrameIndexVideo"]))
                # print "Number of frames", len(annotation)
                ret, frame = cap.read()
                if (not ret):
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_index += 1

                bodypart_coords_gt[videoNumber][frameNumber] = {}
                bodypart_coords_est[videoNumber][frameNumber] = {}
                bodypart_coords_interp[videoNumber][frameNumber] = {}

                distance_coord_gt_interp[videoNumber][frameNumber] = {}
                distance_coord_gt_estim[videoNumber][frameNumber] = {}
                distance_coord_estim_interp[videoNumber][frameNumber] = {}

                if (options.display_level >= 2):
                    display_voters = frame.copy()

                for j in range(0, len(annotation[frameNumber]["FrameValueCoordinates"])):
                    if (annotation[frameNumber]["FrameValueCoordinates"][j]["Name"] == test_bodypart):
                        bodypart_coords_gt[videoNumber][frameNumber]["x"] = int(
                            annotation[frameNumber]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                        bodypart_coords_gt[videoNumber][frameNumber]["y"] = int(
                            annotation[frameNumber]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])
                        bodypart_coords_gt[videoNumber][frameNumber]["FrameIndexVideo"] = int(
                            annotation[frameNumber]["FrameIndexVideo"])
                        bodypart_coords_gt[videoNumber][frameNumber]["BodyPart"] = test_bodypart

                if ( bodypart_coords_gt[videoNumber][frameNumber] == {} ):
                    print "Empty Image Video %d, Frame %d" % (videoNumber, frameNumber)
                    continue

                print "frame_index:", frame_index

                bodypart_coords_est[videoNumber][frameNumber]["x"] = bodypart_coords_gt[videoNumber][frameNumber]["x"]
                bodypart_coords_est[videoNumber][frameNumber]["y"] = bodypart_coords_gt[videoNumber][frameNumber]["y"]
                bodypart_coords_est[videoNumber][frameNumber]["conf"] = 0
                bodypart_coords_est[videoNumber][frameNumber]["FrameIndexVideo"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["FrameIndexVideo"]
                bodypart_coords_est[videoNumber][frameNumber]["BodyPart"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["BodyPart"]

                bodypart_coords_interp[videoNumber][frameNumber]["x"] = bodypart_coords_gt[videoNumber][frameNumber][
                    "x"]
                bodypart_coords_interp[videoNumber][frameNumber]["y"] = bodypart_coords_gt[videoNumber][frameNumber][
                    "y"]
                bodypart_coords_interp[videoNumber][frameNumber]["FrameIndexVideo"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["FrameIndexVideo"]
                bodypart_coords_interp[videoNumber][frameNumber]["BodyPart"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["BodyPart"]

                distance_coord_gt_interp[videoNumber][frameNumber]["BodyPart"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["BodyPart"]
                distance_coord_gt_interp[videoNumber][frameNumber]["FrameIndexVideo"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["FrameIndexVideo"]

                distance_coord_gt_estim[videoNumber][frameNumber]["BodyPart"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["BodyPart"]
                distance_coord_gt_estim[videoNumber][frameNumber]["FrameIndexVideo"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["FrameIndexVideo"]

                distance_coord_estim_interp[videoNumber][frameNumber]["BodyPart"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["BodyPart"]
                distance_coord_estim_interp[videoNumber][frameNumber]["FrameIndexVideo"] = \
                    bodypart_coords_gt[videoNumber][frameNumber]["FrameIndexVideo"]

                # perform detection
                images = []
                packed_image_headers = []
                images_data = []
                start_time = time.clock()
                image = copy.deepcopy(frame)

                crop_x = 0
                crop_y = 0
                # image = image[crop_y:1920,crop_x:1920,0]

                images.append(copy.deepcopy(image))

                image_info = np.shape(image)

                if not image_info:
                    print "Empty Image Video %d, Frame %d" % (videoNumber, frameNumber)
                    continue

                image_header = ('01', image_info[0], image_info[1], crop_x, crop_y)
                packed_image_header = packer_image_header.pack(*image_header)
                packed_image_headers.append(copy.deepcopy(packed_image_header))

                image_data = np.getbuffer(np.ascontiguousarray(image))
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
                            # print "trying to send packet to server", s
                            HOST, PORT = "localhost", 9988 + s
                            if ( socks[s] == None ):
                                socks[s] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                            socks[s].connect((HOST, PORT))
                            # print "Host Connected..... on PORT: %d" % (PORT)
                            socks[s].setblocking(1)
                            # print "PORT %d Blocked" % (PORT)
                            socks[s].sendall(packed_header)
                            # print "Sending Header....."
                            socks[s].sendall(packed_list_header)
                            # print "Sending Header List....."
                            # cv2.waitKey(5000)

                            for j in range(0, len(images_data)):
                                resl = socks[s].sendall(packed_image_headers[j])
                                socks[s].sendall(images_data[j])
                                # if resl == None:
                                #     print "Packet Sent Successfully....."

                            # print "Sent packet on server", s
                            server_free[s] = False
                        except:
                            pass
                        finally:
                            break

                for s in range(0, n_server):
                    if ( not server_free[s] ):
                        # Receive data from the server and shut down
                        try:
                            # socks[s].settimeout(1/20.0)
                            received = socks[s].recv(unpacker_ack_header.size)
                            ack_header = unpacker_ack_header.unpack(received)
                            # print "received a packet from server; packet header:", ack_header[0]
                            received = socks[s].recv(ack_header[1])
                            # received = socks[s].recv(1024)
                            socks[s].close()
                            socks[s] = None
                            server_free[s] = True
                            # print "Received from server {} : {}".format(s, received)
                            received_json = json.loads(received)

                            if ("detections" in received_json):
                                for di in range(0, len(received_json["detections"])):
                                    if ( received_json["detections"][di]["test_bodypart"] == test_bodypart ):
                                        # print "Writing received values...."
                                        bodypart_coords_est[videoNumber][frameNumber]["x"] = \
                                            received_json["detections"][di]["coord_x"]
                                        bodypart_coords_est[videoNumber][frameNumber]["y"] = \
                                            received_json["detections"][di]["coord_y"]
                                        bodypart_coords_est[videoNumber][frameNumber]["conf"] = received_json["detections"][di]["conf"]
                                        # print "Received Confidence:", bodypart_coords_est[videoNumber][frameNumber]["conf"]
                                    else:
                                        bodypart_coords_est[videoNumber][frameNumber]["x"] = -1
                                        bodypart_coords_est[videoNumber][frameNumber]["y"] = -1
                        except socket.timeout:
                            pass
                        except:
                            print "Unexpected error:", sys.exc_info()[0]

                        if bodypart_coords_gt[videoNumber][frameNumber]["x"] != -1:
                            distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] = np.sqrt(np.square(
                                bodypart_coords_gt[videoNumber][frameNumber]["x"] -
                                bodypart_coords_est[videoNumber][frameNumber]["x"]) +
                                                                                                    np.square(
                                                                                                        bodypart_coords_gt[
                                                                                                            videoNumber][
                                                                                                            frameNumber][
                                                                                                            "y"] -
                                                                                                        bodypart_coords_est[
                                                                                                            videoNumber][
                                                                                                            frameNumber][
                                                                                                            "y"]))
                        else:
                            distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] = None

                        c = bodypart_coords_est[videoNumber][frameNumber]["conf"]
                        # print "Confidence: ", c
                        confidence_votes.append(c)
                        if distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 4:
                            confidence_votes_4.append(c)
                        elif distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 3:
                            confidence_votes_3.append(c)
                        elif distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 2:
                            confidence_votes_2.append(c)
                        elif distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 5:
                            confidence_votes_5.append(c)
                        elif distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 6:
                            confidence_votes_6.append(c)
                        elif distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 7:
                            confidence_votes_7.append(c)
                        elif distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 1:
                            confidence_votes_1.append(c)
                        elif distance_coord_gt_estim[videoNumber][frameNumber]["Distance"] <= 8:
                            confidence_votes_8.append(c)
                        elif bodypart_coords_gt[videoNumber][frameNumber]["x"] == -1:
                            confidence_votes_None.append(c)

                # print server_free
                end_time = time.clock()
                dif = end_time - start_time
                total_time_frame.append(dif)
                # print "Time Taken: ", dif

        else:
            print "Not able to read video file"

        print "Length: ",len(bodypart_coords_gt[videoNumber])
        for m in range(0, len(bodypart_coords_gt[videoNumber]) - 3):
            if (m > 0 and m < ((len(bodypart_coords_gt[videoNumber]) - 4))):
                if ((bodypart_coords_gt[videoNumber][m - 1]["x"] != -1) and (bodypart_coords_gt[videoNumber][m + 1]["x"] != -1) and (bodypart_coords_gt[videoNumber][m]["x"] != -1)):
                    bodypart_coords_interp[videoNumber][m]["x"] = int(
                        (bodypart_coords_gt[videoNumber][m - 1]["x"] + bodypart_coords_gt[videoNumber][m + 1]["x"]) / 2)
                    bodypart_coords_interp[videoNumber][m]["y"] = int(
                        (bodypart_coords_gt[videoNumber][m - 1]["y"] + bodypart_coords_gt[videoNumber][m + 1]["y"]) / 2)
                    distance_coord_gt_interp[videoNumber][m]["Distance"] = np.sqrt(np.square(
                        bodypart_coords_gt[videoNumber][m]["x"] - bodypart_coords_interp[videoNumber][m][
                            "x"]) + np.square(
                        bodypart_coords_gt[videoNumber][m]["y"] - bodypart_coords_interp[videoNumber][m]["y"]))
                    distance_coord_estim_interp[videoNumber][m]["Distance"] = np.sqrt(np.square(
                        bodypart_coords_est[videoNumber][m]["x"] - bodypart_coords_interp[videoNumber][m][
                            "x"]) + np.square(
                        bodypart_coords_est[videoNumber][m]["y"] - bodypart_coords_interp[videoNumber][m]["y"]))
                else:
                    bodypart_coords_interp[videoNumber][m]["x"] = int(bodypart_coords_gt[videoNumber][m]["x"])
                    bodypart_coords_interp[videoNumber][m]["y"] = int(bodypart_coords_gt[videoNumber][m]["y"])
                    distance_coord_gt_interp[videoNumber][m]["Distance"] = None
                    distance_coord_estim_interp[videoNumber][m]["Distance"] = None
            elif m == 0:
                distance_coord_gt_interp[videoNumber][m]["Distance"] = None
                distance_coord_estim_interp[videoNumber][m]["Distance"] = None
                if bodypart_coords_gt[videoNumber][m]["x"] != -1:
                    bodypart_coords_interp[videoNumber][m]["x"] = int(bodypart_coords_gt[videoNumber][m]["x"])
                    bodypart_coords_interp[videoNumber][m]["y"] = int(bodypart_coords_gt[videoNumber][m]["y"])

            elif (m == len(bodypart_coords_gt[videoNumber]) - 4):
                distance_coord_gt_interp[videoNumber][m]["Distance"] = None
                distance_coord_estim_interp[videoNumber][m]["Distance"] = None
                if bodypart_coords_gt[videoNumber][m]["x"] != -1:
                    bodypart_coords_interp[videoNumber][m]["x"] = int(bodypart_coords_gt[videoNumber][m]["x"])
                    bodypart_coords_interp[videoNumber][m]["y"] = int(bodypart_coords_gt[videoNumber][m]["y"])

        tempDist_vector_gt_interp = []
        tempDist_vector_gt_estim = []
        tempDist_vector_estim_interp = []
        timeStats["Time_All_Frames"] = total_time_frame
        timeStats["Time_Mean"] = np.mean(total_time_frame)

        for i in range(0, len(distance_coord_gt_interp[videoNumber]) - 3):
            if distance_coord_gt_interp[videoNumber][i]["Distance"] != None:
                tempDist_vector_gt_interp.append(distance_coord_gt_interp[videoNumber][i]["Distance"])
                tempDist_vector_gt_interp_tot.append(distance_coord_gt_interp[videoNumber][i]["Distance"])
            if distance_coord_gt_estim[videoNumber][i]["Distance"] != None:
                tempDist_vector_gt_estim.append(distance_coord_gt_estim[videoNumber][i]["Distance"])
                tempDist_vector_gt_estim_tot.append(distance_coord_gt_estim[videoNumber][i]["Distance"])
            if distance_coord_estim_interp[videoNumber][i]["Distance"] != None:
                tempDist_vector_estim_interp.append(distance_coord_estim_interp[videoNumber][i]["Distance"])
                tempDist_vector_estim_interp_tot.append(distance_coord_estim_interp[videoNumber][i]["Distance"])

        distance_coord_gt_interp[videoNumber]["MeanDistance"] = np.mean(tempDist_vector_gt_interp)
        distance_coord_gt_interp[videoNumber]["MedianDistance"] = np.median(tempDist_vector_gt_interp)

        distance_coord_estim_interp[videoNumber]["MeanDistance"] = np.mean(tempDist_vector_estim_interp)
        distance_coord_estim_interp[videoNumber]["MedianDistance"] = np.median(tempDist_vector_estim_interp)

        distance_coord_gt_estim[videoNumber]["MeanDistance"] = np.mean(tempDist_vector_gt_estim)
        distance_coord_gt_estim[videoNumber]["MedianDistance"] = np.median(tempDist_vector_gt_estim)

    json.dump(bodypart_coords_gt, fileWriter_annotatedPosition, sort_keys=True, indent=4, separators=(',', ': '))
    json.dump(bodypart_coords_interp, fileWriter_interpolatedPosition, sort_keys=True, indent=4, separators=(',', ': '))
    json.dump(bodypart_coords_est, fileWriter_estimatedPosition, sort_keys=True, indent=4, separators=(',', ': '))

    json.dump(distance_coord_gt_interp, fileWriter_distAnnotInterp, sort_keys=True, indent=4, separators=(',', ': '))
    json.dump(distance_coord_estim_interp, fileWriter_distInterpEstim, sort_keys=True, indent=4, separators=(',', ': '))
    json.dump(distance_coord_gt_estim, fileWriter_distAnnotEstim, sort_keys=True, indent=4, separators=(',', ': '))

    all_Means_Medians = {}
    all_Means_Medians["Annotated_and_Interpolated"] = {}
    all_Means_Medians["Estimated_and_Interpolated"] = {}
    all_Means_Medians["Annotated_and_Estimated"] = {}
    all_Means_Medians["Confidence"] = {}

    all_Means_Medians["Annotated_and_Interpolated"]["MeanDistance"] = np.mean(tempDist_vector_gt_interp_tot)
    all_Means_Medians["Annotated_and_Interpolated"]["MedianDistance"] = np.median(tempDist_vector_gt_interp_tot)
    all_Means_Medians["Estimated_and_Interpolated"]["MeanDistance"] = np.mean(tempDist_vector_estim_interp_tot)
    all_Means_Medians["Estimated_and_Interpolated"]["MedianDistance"] = np.median(tempDist_vector_estim_interp_tot)
    all_Means_Medians["Annotated_and_Estimated"]["MeanDistance"] = np.mean(tempDist_vector_gt_estim_tot)
    all_Means_Medians["Annotated_and_Estimated"]["MedianDistance"] = np.median(tempDist_vector_gt_estim_tot)
    all_Means_Medians["Confidence"]["Mean"] = np.mean(confidence_votes)
    all_Means_Medians["Confidence"]["Median"] = np.median(confidence_votes)

    confidence_votes_distribution["1"] = confidence_votes_1
    confidence_votes_distribution["2"] = confidence_votes_2
    confidence_votes_distribution["3"] = confidence_votes_3
    confidence_votes_distribution["4"] = confidence_votes_4
    confidence_votes_distribution["5"] = confidence_votes_5
    confidence_votes_distribution["6"] = confidence_votes_6
    confidence_votes_distribution["7"] = confidence_votes_7
    confidence_votes_distribution["8"] = confidence_votes_8
    confidence_votes_distribution["None"] = confidence_votes_None

    json.dump(all_Means_Medians, fileWriter_allMeans, sort_keys=True, indent=4, separators=(',', ': '))
    json.dump(timeStats, fileWriter_allTime, sort_keys=True, indent=4, separators=(',', ': '))
    json.dump(confidence_votes_distribution, fileWriter_confidence, sort_keys=True, indent=4, separators=(',', ': '))

    fileWriter_annotatedPosition.close()
    fileWriter_estimatedPosition.close()
    fileWriter_interpolatedPosition.close()
    fileWriter_distAnnotInterp.close()
    fileWriter_distAnnotEstim.close()
    fileWriter_distInterpEstim.close()
    fileWriter_allMeans.close()
    fileWriter_confidence.close()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",
                      help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,
                      help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",
                      help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--test-bodypart", dest="test_bodypart", default="MouthHook",
                      help="Input the bodypart to be tested")
    parser.add_option("", "--save-folder", dest="save_folder", default="")

    (options, args) = parser.parse_args()

    main(options, args)
