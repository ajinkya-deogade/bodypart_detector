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
    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.extend(test_annotation["Annotations"])

    print "len(test_annotations):", len(test_annotations)

    frame_index = -1

    error_stats_all = []

    bodypart_gt = {}

    for j in range(0, len(test_annotations)):
        frame_index += 1

        annotation = test_annotations[j]

        frame_file = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        frame = cv2.imread(frame_file)
        if (options.display_level >= 2):
            display_voters = frame.copy()
            
        bodypart_coords_gt = None
        for j in range(0, len(annotation["FrameValueCoordinates"])):
            if (annotation["FrameValueCoordinates"][j]["Name"] == test_bodypart and annotation["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt = {}
                bodypart_coords_gt["x"] = int(annotation["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                bodypart_coords_gt["y"] = int(annotation["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        if ( bodypart_coords_gt == None ):
            continue

        print "frame_index:", frame_index
    
        bodypart_gt[frame_index] = {}
        bodypart_gt[frame_index]["bodypart_coords_gt"] = bodypart_coords_gt
        bodypart_gt[frame_index]["frame_file"] = frame_file

        # perform detection
        images = []
        packed_image_headers = []
        images_data = []

        image = copy.deepcopy(frame)

        print "bodypart_coords_gt:" , bodypart_coords_gt
        crop_x = 0
        crop_y = 0
        image = image[crop_y:1920,crop_x:1920,0]
        
        images.append(copy.deepcopy(image))

        image_info = np.shape(image)
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
                
                    HOST, PORT = "localhost", 9989 + s
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
                bodypart_coords_est = None
                try:
                    socks[s].settimeout(1/200.0)
                    received = socks[s].recv(1024)
                    socks[s].close()
                    socks[s] = None
                    server_free[s] = True
                    print "Received from server {} : {}".format(s, received)
                    received_json = json.loads(received)
                
                    if ( "detections" in received_json ):
                        for di in range(0, len(received_json["detections"])):
                            if ( received_json["detections"][di]["test_bodypart"] == test_bodypart ):
                                bodypart_coords_est = {}
                                bodypart_coords_est["x"] = received_json["detections"][di]["coord_x"]
                                bodypart_coords_est["y"] = received_json["detections"][di]["coord_y"]
                                f_index = received_json["detections"][di]["frame_index"]

                except:
                    pass

                if (bodypart_coords_est is not None):
                    error_stats = Error_Stats()
                    error_stats.frame_file =  bodypart_gt[f_index]["frame_file"]
                    error_stats.error_distance = np.sqrt(np.square(bodypart_gt[f_index]["bodypart_coords_gt"]["x"] - bodypart_coords_est["x"]) + 
                                                         np.square(bodypart_gt[f_index]["bodypart_coords_gt"]["y"] - bodypart_coords_est["y"]))
                    print bodypart_gt[f_index]["frame_file"], "Distance between annotated and estimated RightMHhook location:", error_stats.error_distance
                    error_stats_all.append(error_stats)

        print server_free

    print os.sys.argv
    error_distance_inliers = []
    for es in error_stats_all:
        if (es.error_distance <= options.outlier_error_dist):
            error_distance_inliers.append(es.error_distance)
    n_outlier = len(error_stats_all) - len(error_distance_inliers)
    print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (options.outlier_error_dist, n_outlier, len(error_stats_all), float(n_outlier) / float(max(1, float(len(error_stats_all)))))
    print "Median inlier error dist =", np.median(error_distance_inliers)
    print "Mean inlier error dist =", np.mean(error_distance_inliers)
    print "len(bodypart_gt):", len(bodypart_gt)

    cv2.destroyWindow("frame")



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--test-bodypart", dest="test_bodypart",default="MouthHook", help="Input the bodypart to be tested")

    (options, args) = parser.parse_args()

    main(options, args)
