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
import csv

packer_header = struct.Struct('= 2s I')
packer_list_header = struct.Struct('= 2s I I')
packer_image_header = struct.Struct('= 2s I I I I')

packer_keypoint_header = struct.Struct('= 2s I I') # Keypoint Information: Rows x Columns
packer_list_keypoint_header = struct.Struct('= 2s I I') # Keypoint Information: FrameNumber and  DataSize
packer_keypoint_data = struct.Struct('= 2s d')  # Keypoint Information: KeypointData

packer_descriptor_header = struct.Struct('= 2s I I') # Descriptor Information: Rows x Columns
packer_list_descriptor_header = struct.Struct('= 2s I I') # Descriptor Information: FrameNumber and  DataSize
packer_descriptor_data = struct.Struct('= 2s I')  # Descriptor Information: DescriptorData

unpacker_ack_header = struct.Struct('= 2s I')

class Error_Stats:
    def __init__(self):
        self.frame_file = None


def main(options, args):
    socks = []
    server_free = []
    n_server = options.n_server
    for i in range(0, n_server):
        socks.append( socket.socket(socket.AF_INET, socket.SOCK_STREAM) )
        socks[i].connect(('localhost', options.port + i))
        server_free.append(True)

    test_bodypart = options.test_bodypart

    test_annotations = []
    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.extend(test_annotation["Annotations"])

    print "len(test_annotations):" , len(test_annotations)

    frame_index = -1

    error_stats_all = []

    bodypart_gt = {}
    n_frame_sent_to_server = 0
    crop_size = options.crop_size

#    for j in range(10, 11):
    for j in range(0, len(test_annotations)):
        # os.system('clear')
        frame_index += 1

        annotation = test_annotations[j]

        frame_file = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        frame = cv2.imread(frame_file)
        if (options.display_level >= 2):
            display_voters = frame.copy()
            
        bodypart_coords_gt = None
        bodypart_coords_mh = None
        for k in range(0, len(annotation["FrameValueCoordinates"])):
            if (annotation["FrameValueCoordinates"][k]["Name"] == test_bodypart and annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"] != -1):
                bodypart_coords_gt = {}
                bodypart_coords_gt["x"] = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                bodypart_coords_gt["y"] = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])
            if (annotation["FrameValueCoordinates"][k]["Name"] == "MouthHook" and annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"] != -1):
                bodypart_coords_mh = {}
                bodypart_coords_mh["x"] = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                bodypart_coords_mh["y"] = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])

        if ( bodypart_coords_gt == None ):
            continue

        if ( options.verbosity >= 1 ):
            print "frame_index:", frame_index
    
        bodypart_gt[frame_index] = {}
        bodypart_gt[frame_index]["bodypart_coords_gt"] = bodypart_coords_gt
        bodypart_gt[frame_index]["frame_file"] = frame_file

#        for iteration in range(0, 1000):
        for iteration in range(0, 1):
            # perform detection
            images = []
            packed_image_headers = []
            images_data = []

            ## Read and Pack Image Data
            image = copy.deepcopy(frame)
            
            if ( options.verbosity >= 1 ):
                print "bodypart_coords_gt:" , bodypart_coords_gt

            crop_x = max(0, bodypart_coords_mh["x"]-100)
            crop_y = max(0, bodypart_coords_mh["y"]-100)
            image = image[crop_y:crop_y+crop_size,crop_x:crop_x+crop_size,0]
            
            images.append(copy.deepcopy(image))
            
            image_info = np.shape(image)
            if ( options.verbosity >= 1 ):
                print "Image Info: ", image_info
            image_header = ('01', image_info[0], image_info[1], crop_x, crop_y)
            packed_image_header = packer_image_header.pack(*image_header)

            packed_image_headers.append(copy.deepcopy(packed_image_header))
            # packed_image_headers.append(copy.deepcopy(packed_image_header))
            
            image_data = np.getbuffer(np.ascontiguousarray(image))
            if ( options.verbosity >= 1 ):
                print "Length pf Image Buffer: ", len(image_data)
            images_data.append(image_data)
            # images_data.append(image_data)

            # if ( options.verbosity >= 1 ):
            #     print len(images_data)

            image_list_header = ('00', frame_index, len(images_data))
            packed_image_list_header = packer_list_header.pack(*image_list_header)

            blob_size = len(packed_image_list_header)
            for k in range(0, len(images_data)):
                blob_size += len(packed_image_headers[k]) + len(images_data[k])
            
            image_blob_header = ('01', blob_size)
            packed_image_blob_header = packer_header.pack(*image_blob_header)

            ## Read and Pack Keypoints Data
            current_dir = os.path.abspath(os.path.dirname(frame_file))
            parent_dir = os.path.basename(current_dir)
            keypoints_folder = os.path.join(options.dir_keypoints, parent_dir)

            if not os.path.exists(keypoints_folder):
                print "Folder does not exist !!!"

            keypoints_file = os.path.join(keypoints_folder, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"

            keypoints = []
            with open(keypoints_file, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
                    row = row[0]
                    temp_feature = []
                    temp_feature.append(float(row[0]))
                    temp_feature.append(float(row[1]))
                    temp_feature.append(float((float(row[2])/float(1.2))*float(9)))
                    temp_feature.append(float(row[6]))
                    temp_feature.append(float(row[3]))
                    temp_feature.append(float(row[4]))
                    temp_feature.append(float(row[5]))
                    keypoints.append(temp_feature)

            if options.verbosity > 0:
                print "Keypoint Info: ", np.shape(keypoints)

            keypoint_info = np.shape(keypoints)
            keypoint_header = ('01', keypoint_info[0], keypoint_info[1])
            packed_keypoint_header = packer_keypoint_header.pack(*keypoint_header)

            keypoint_data = np.getbuffer(np.ascontiguousarray(keypoints, dtype=np.float32))
            if options.verbosity > 0:
                print "Keypoint Buffer: ", len(keypoint_data)
            keypoint_list_header = ('00', frame_index, len(keypoint_data))
            packed_keypoint_list_header = packer_list_keypoint_header.pack(*keypoint_list_header)

            blob_size_kp = len(packed_keypoint_list_header)
            blob_size_kp += len(packed_keypoint_header) + len(keypoint_data)

            kp_blob_header = ('01', blob_size_kp)
            packed_kp_blob_header = packer_keypoint_data.pack(*kp_blob_header)

            ## Read and Pack Descriptor Data
            current_dir = os.path.abspath(os.path.dirname(frame_file))
            parent_dir = os.path.basename(current_dir)
            descriptors_folder = os.path.join(options.dir_descriptors, parent_dir)

            if not os.path.exists(descriptors_folder):
                print "Folder does not exist !!!"

            descriptors_file = os.path.join(descriptors_folder, os.path.splitext(os.path.basename(annotation["FrameFile"]))[0]) + ".csv"

            descriptors = []
            with open(descriptors_file, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    row = [[float(val) for val in ro.strip().split('\t')] for ro in row]
                    row = row[0]
                    # print "Shape of the row: ", np.shape(row)
                    descriptors.append(row)

            if options.verbosity > 0:
                print "Keypoint Info: ", np.shape(keypoints)

            descriptor_info = np.shape(descriptors)
            descriptor_header = ('01', descriptor_info[0], descriptor_info[1])
            packed_descriptor_header = packer_descriptor_header.pack(*descriptor_header)

            descriptor_data = np.getbuffer(np.ascontiguousarray(descriptors, dtype=np.float32))
            if options.verbosity > 0:
                print "Keypoint Buffer: ", len(keypoint_data)

            descriptor_list_header = ('00', frame_index, len(descriptor_data))
            packed_descriptor_list_header = packer_list_descriptor_header.pack(*descriptor_list_header)

            blob_size_desc = len(packed_descriptor_list_header)
            blob_size_desc += len(packed_descriptor_header) + len(descriptor_data)

            desc_blob_header = ('01', blob_size_desc)
            packed_desc_blob_header = packer_descriptor_data.pack(*desc_blob_header)

            received_json = None
        
            for s in range(0, n_server):
                if ( server_free[s] ):
                    try:
                        if ( options.verbosity >= 1 ):
                            print "trying to send packet to server", s
                        
                        socks[s].setblocking(1)

                        ## Send the Image Data to the Server
                        socks[s].sendall(packed_image_blob_header)
                        socks[s].sendall(packed_image_list_header)
                        for k in range(0, len(images_data)):
                            socks[s].sendall(packed_image_headers[k])
                            socks[s].sendall(images_data[k])

                        ## Send a GAP String between Image and Keypoints
                        gap_kp = struct.pack("s", "kp")
                        socks[s].sendall(gap_kp)

                        ## Send the Keypoint Data to the Server
                        socks[s].sendall(packed_kp_blob_header)
                        socks[s].sendall(packed_keypoint_list_header)
                        socks[s].sendall(packed_keypoint_header)
                        socks[s].sendall(keypoint_data)

                        ## Send a GAP String between Keypoints and Descriptors
                        gap_desc = struct.pack("s", "desc")
                        socks[s].sendall(gap_desc)

                        ## Send the Descriptor Data to the Server
                        socks[s].sendall(packed_desc_blob_header)
                        socks[s].sendall(packed_descriptor_list_header)
                        socks[s].sendall(packed_descriptor_header)
                        socks[s].sendall(descriptor_data)

                        if ( options.verbosity >= 1 ):
                            print "Sent packet on server", s

                        server_free[s] = False
                        n_frame_sent_to_server += 1
                    except:
                        print "Unexpected error while sending:", sys.exc_info()[0]
                        return
                    finally:
                        break

            for s in range(0, n_server):
                if ( not server_free[s] ):
                    # Receive data from the server
                    bodypart_coords_est = None
                    try:
                        # socks[s].settimeout(1/1000.0)
                        received = socks[s].recv(unpacker_ack_header.size)
                        ack_header = unpacker_ack_header.unpack(received)
                        if ( options.verbosity >= 1 ):
                            print "received a packet from server; packet header:", ack_header[0]
                        received = socks[s].recv(ack_header[1])
                        server_free[s] = True
                        if ( options.verbosity >= 1 ):
                            print "Received from server {} : {}".format(s, received)
                        received_json = json.loads(received)
                        
                        if ( "detections" in received_json ):
                            for di in range(0, len(received_json["detections"])):
                                if ( "test_bodypart" in received_json["detections"][di] and received_json["detections"][di]["test_bodypart"] == test_bodypart ):
                                    bodypart_coords_est = {}
                                    bodypart_coords_est["x"] = received_json["detections"][di]["coord_x"]
                                    bodypart_coords_est["y"] = received_json["detections"][di]["coord_y"]
                                    bodypart_coords_est["conf"] = received_json["detections"][di]["conf"]
                                    bodypart_coords_est["frame_index"] = received_json["detections"][di]["frame_index"]
                    # except socket.timeout:
                    #     pass
                    except:
                        print "Unexpected error while receiving:", sys.exc_info()[0]

                    if (bodypart_coords_est is not None):
                        error_stats = Error_Stats()
                        error_stats.frame_file =  bodypart_gt[bodypart_coords_est["frame_index"]]["frame_file"]
                        error_stats.error_distance = np.sqrt(np.square(bodypart_gt[bodypart_coords_est["frame_index"]]["bodypart_coords_gt"]["x"] - bodypart_coords_est["x"]) + 
                                                             np.square(bodypart_gt[bodypart_coords_est["frame_index"]]["bodypart_coords_gt"]["y"] - bodypart_coords_est["y"]))
                        error_stats.conf = bodypart_coords_est["conf"]
                        if ( options.verbosity >= 1 ):
                            print bodypart_gt[bodypart_coords_est["frame_index"]]["frame_file"], "Distance between annotated and estimated RightMHhook location:", error_stats.error_distance
                        error_stats_all.append(error_stats)

                if ( options.verbosity >= 1 ):
                    print server_free

    print os.sys.argv
    error_distance_inliers = []
    inlier_confs = []
    outlier_confs = []
    for es in error_stats_all:
        if (es.error_distance <= options.outlier_error_dist):
            error_distance_inliers.append(es.error_distance)
            inlier_confs.append(es.conf)
        else:
            outlier_confs.append(es.conf)

    n_outlier = len(error_stats_all) - len(error_distance_inliers)

    print "Total number of frames sent to server:", n_frame_sent_to_server
    print "Total number of detections received:", len(error_stats_all)
    print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (options.outlier_error_dist, n_outlier, len(error_stats_all), float(n_outlier) / max(1, float(len(error_stats_all))) )
    if ( len(error_distance_inliers) > 0 ):
        print "Median inlier error dist =", np.median(error_distance_inliers)
        print "Mean inlier error dist =", np.mean(error_distance_inliers)
        print "Min inlier confidence =", np.min(inlier_confs)
        print "Mean inlier confidence =", np.mean(inlier_confs)
    if ( len(outlier_confs) > 0 ):
        print "Max outlier confidence =", np.max(outlier_confs)
        print "Mean outlier confidence =", np.mean(outlier_confs)
    print "len(bodypart_gt):", len(bodypart_gt)

    cv2.destroyWindow("frame")



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--n-server", dest="n_server", default=1, type="int",help="number of servers available")
    parser.add_option("", "--test-bodypart", dest="test_bodypart",default="MouthHook", help="Input the bodypart to be tested")
    parser.add_option("", "--verbosity", dest="verbosity", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--dir-keypoints", dest="dir_keypoints", default="", help="directory to save keypoints")
    parser.add_option("", "--dir-descriptors", dest="dir_descriptors", default="", help="directory to save keypoints")
    parser.add_option("", "--crop-size", dest="crop_size", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--port-number", dest="port", type="int", default=0, help="degree of verbosity")

    (options, args) = parser.parse_args()

    main(options, args)
