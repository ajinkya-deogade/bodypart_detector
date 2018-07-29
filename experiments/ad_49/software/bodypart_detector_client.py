#! /usr/bin/env python

import copy
import json
import os
import re
import socket
import struct
import sys
from optparse import OptionParser

import cv2
import numpy as np

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
    n_server = options.n_server
    for i in range(0, n_server):
        socks.append( socket.socket(socket.AF_INET, socket.SOCK_STREAM) )
        socks[i].connect(('', 9998 + i))
        server_free.append(True)

    test_annotations = []
    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.extend(test_annotation["Annotations"])

    print "len(test_annotations):" , len(test_annotations)

    frame_index = -1

    error_stats_all = {}
    n_instance_gt = {}
    n_instance_gt["MouthHook"] = 0
    for s in options.detect_bodypart:
        error_stats_all[s] = []
        n_instance_gt[s] = 0
        
    bodypart_gt = {}
    n_frame_sent_to_server = 0

#    for j in range(10, 11):
    for j in range(0, len(test_annotations)):
        frame_index += 1

        annotation = test_annotations[j]

        frame_file = annotation["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        frame = cv2.imread(frame_file)
        if (options.display_level >= 2):
            display_voters = frame.copy()

        flag_skip = True
        bodypart_coords_gt = {}
        for k in range(0, len(annotation["FrameValueCoordinates"])):
            bi = annotation["FrameValueCoordinates"][k]["Name"]
            if ((bi == "MouthHook" or any(bi == s for s in options.detect_bodypart)) and annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"] != -1 and annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"] != -1):
                flag_skip = False
                bodypart_coords_gt[bi] = {}
                bodypart_coords_gt[bi]["x"] = int(annotation["FrameValueCoordinates"][k]["Value"]["x_coordinate"])
                bodypart_coords_gt[bi]["y"] = int(annotation["FrameValueCoordinates"][k]["Value"]["y_coordinate"])
                n_instance_gt[bi] += 1

        if ( flag_skip ):
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
            
            image = copy.deepcopy(frame)
            
            if ( options.verbosity >= 1 ):
                print "bodypart_coords_gt:" , bodypart_coords_gt
            crop_x = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["x"]-100)
            crop_y = max(0, bodypart_gt[frame_index]["bodypart_coords_gt"]["MouthHook"]["y"]-100)
            image = image[crop_y:crop_y+200,crop_x:crop_x+200,0]
            
            images.append(copy.deepcopy(image))
            
            image_info = np.shape(image)
            if ( options.verbosity >= 1 ):
                print image_info
            image_header = ('01', image_info[0], image_info[1], crop_x, crop_y)
            packed_image_header = packer_image_header.pack(*image_header)
            packed_image_headers.append(copy.deepcopy(packed_image_header))
            packed_image_headers.append(copy.deepcopy(packed_image_header))
            
            image_data = np.getbuffer(np.ascontiguousarray(image))
            if ( options.verbosity >= 1 ):
                print len(image_data)
            images_data.append(image_data)
            images_data.append(image_data)
            if ( options.verbosity >= 1 ):
                print len(images_data)

            list_header = ('00', frame_index, len(images_data))
            packed_list_header = packer_list_header.pack(*list_header)

            blob_size = len(packed_list_header)
            for k in range(0, len(images_data)):
                blob_size += len(packed_image_headers[k]) + len(images_data[k])
            
            header = ('01', blob_size)
            packed_header = packer_header.pack(*header)

            received_json = None
        
            for s in range(0, n_server):
                if ( server_free[s] ):
                    try:
                        if ( options.verbosity >= 1 ):
                            print "trying to send packet to server", s
                        
                        socks[s].setblocking(1)
                        
                        socks[s].sendall(packed_header)
                        
                        socks[s].sendall(packed_list_header)
                        
                        for k in range(0, len(images_data)):
                            socks[s].sendall(packed_image_headers[k])
                            socks[s].sendall(images_data[k])
                        
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
                                tbp = received_json["detections"][di]["test_bodypart"]
                                fi = received_json["detections"][di]["frame_index"]
                                if ( tbp in bodypart_gt[fi]["bodypart_coords_gt"] ):
                                    error_stats = Error_Stats()
                                    error_stats.frame_file =  bodypart_gt[fi]["frame_file"]
                                    error_stats.error_distance = np.sqrt(np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"] - 
                                                                                   received_json["detections"][di]["coord_x"]) + 
                                                                         np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"] - 
                                                                                   received_json["detections"][di]["coord_y"]))
                                    error_stats.conf = received_json["detections"][di]["conf"]
                                    if ( options.verbosity >= 1 ):
                                        print bodypart_gt[fi]["frame_file"], "Distance between annotated and estimated", tbp, "location:", error_stats.error_distance
                                    error_stats_all[tbp].append(error_stats)
                    except socket.timeout:
                        pass
                    except:
                        print "Unexpected error while receiving:", sys.exc_info()[0]

            if ( options.verbosity >= 1 ):
                print server_free

    are_all_servers_free = False
    while ( not are_all_servers_free ):
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
                            tbp = received_json["detections"][di]["test_bodypart"]
                            fi = received_json["detections"][di]["frame_index"]
                            if ( tbp in bodypart_gt[fi]["bodypart_coords_gt"] ):
                                error_stats = Error_Stats()
                                error_stats.frame_file =  bodypart_gt[fi]["frame_file"]
                                error_stats.error_distance = np.sqrt(np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["x"] - 
                                                                               received_json["detections"][di]["coord_x"]) + 
                                                                     np.square(bodypart_gt[fi]["bodypart_coords_gt"][tbp]["y"] - 
                                                                               received_json["detections"][di]["coord_x"]))
                                error_stats.conf = bodypart_coords_est["conf"]
                                if ( options.verbosity >= 1 ):
                                    print bodypart_gt[fi]["frame_file"], "Distance between annotated and estimated", tbp, "location:", error_stats.error_distance
                                error_stats_all[tbp].append(error_stats)
                except socket.timeout:
                    pass
                except:
                    print "Unexpected error while receiving:", sys.exc_info()[0]

        are_all_servers_free = True
        for s in range(0, n_server):
            if ( not server_free[s] ):
                are_all_servers_free = False
                break

    header = ('02', 0)
    packed_header = packer_header.pack(*header)

    for s in range(0, n_server):
        try:
            socks[s].setblocking(1)
            socks[s].sendall(packed_header)
        except:
            print "Unexpected error while sending:", sys.exc_info()[0]
            return


    print os.sys.argv
    for bid in error_stats_all:
        print "Body part:", bid
        error_distance_inliers = []
        inlier_confs = []
        outlier_confs = []
        for es in error_stats_all[bid]:
            if (es.error_distance <= options.outlier_error_dist):
                error_distance_inliers.append(es.error_distance)
                inlier_confs.append(es.conf)
            else:
                outlier_confs.append(es.conf)

        n_outlier = n_instance_gt[bid] - len(error_distance_inliers)

        print "Ground truth number of instances:", n_instance_gt[bid]
        print "Total number of detections received:", len(error_stats_all)
        print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (options.outlier_error_dist, n_outlier, n_instance_gt[bid], float(n_outlier) / max(1, float(n_instance_gt[bid])) )
        if ( len(error_distance_inliers) > 0 ):
            print "Median inlier error dist =", np.median(error_distance_inliers)
            print "Mean inlier error dist =", np.mean(error_distance_inliers)
            print "Min inlier confidence =", np.min(inlier_confs)
            print "Mean inlier confidence =", np.mean(inlier_confs)
        if ( len(outlier_confs) > 0 ):
            print "Max outlier confidence =", np.max(outlier_confs)
            print "Mean outlier confidence =", np.mean(outlier_confs)

    cv2.destroyWindow("frame")



def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--n-server", dest="n_server", default=1, type="int",help="number of servers available")
    parser.add_option("", "--verbosity", dest="verbosity", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--detect-bodypart", dest="detect_bodypart", default="MouthHook", type="string", help="bodypart to detect [MouthHook]", action="callback", callback=string_split)
    parser.add_option("", "--crop-size", dest="crop_size", default="MouthHook", type="string", help="bodypart to detect [MouthHook]", action="callback", callback=string_split)

    (options, args) = parser.parse_args()

    print options.detect_bodypart

    main(options, args)
