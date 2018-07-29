#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle
import struct
import socket

from multiprocessing import Pool
from multiprocessing import Manager
from threading import Lock

class SaveClass:
    def __init__(self):
        self.votes = None
        self.keypoints = None
        self.descriptors = None
        self.bodypart = None

unpacker_header = struct.Struct('= 2s I')
unpacker_list_header = struct.Struct('= 2s I I')
unpacker_image_header = struct.Struct('= 2s I I I I')

packer_ack_header = struct.Struct('= 2s I')

surf = cv2.SURF(500, nOctaves=2, nOctaveLayers=3)
# surf = cv2.SURF(250, nOctaves=2, nOctaveLayers=3)

test_bodypart = None
bodypart_trained_data_pos = None
bodypart_vote = []
bodypart_knn_pos = cv2.KNearest()
bodypart_knn_neg = cv2.KNearest()

def main(options, args):
    global test_bodypart
    global bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_vote

    bodypart_trained_data_pos = SaveClass()
    bodypart_trained_data_pos = pickle.load(open(options.train_data_p, 'rb'))
    bodypart_trained_data_neg = SaveClass()
    bodypart_trained_data_neg = pickle.load(open(options.train_data_n, 'rb'))

    test_bodypart = bodypart_trained_data_neg.bodypart
    print "test_bodypart:" , test_bodypart

    # bodypart_knn_pos = cv2.flann_Index(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32), dict(algorithm=1, trees=4))
    # bodypart_knn_neg = cv2.flann_Index(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float32), dict(algorithm=1, trees=4))

    bodypart_knn_pos.train(np.array(bodypart_trained_data_pos.descriptors, dtype=np.float32), bodypart_trained_data_pos.keypoints)
    bodypart_knn_neg.train(np.array(bodypart_trained_data_neg.descriptors, dtype=np.float32), bodypart_trained_data_neg.keypoints)

    bodypart_vote = np.zeros((2 * options.vote_patch_size + 1, 2 * options.vote_patch_size + 1, 1), np.float)
    
    for x in range(-options.vote_patch_size, options.vote_patch_size + 1):
        for y in range(-options.vote_patch_size, options.vote_patch_size + 1):
            bodypart_vote[y + options.vote_patch_size, x + options.vote_patch_size] = 1.0 + np.exp(
                -0.5 * (x * x + y * y) / (np.square(options.vote_sigma))) / (options.vote_sigma * np.sqrt(2 * np.pi))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", options.socket_port))
    sock.listen(1)
    conn, addr = sock.accept()
    conn.setblocking(1)
    print 'Connected by', addr
    while 1:
        # os.system('clear')
        data = conn.recv(unpacker_header.size)
        header = unpacker_header.unpack(data)
        packet = {}
        packet["version"] = header[0]

        if ( packet["version"] == "02" ):
            break

        assert packet["version"] == "01"
        packet["size"] = header[1]
        if ( options.verbosity >= 1 ):
            print "packet version:" , packet["version"]
            print "packet size:" , packet["size"]

        data = conn.recv(unpacker_list_header.size)
        list_header = unpacker_list_header.unpack(data)
        packet["list"] = {}
        packet["list"]["version"] = list_header[0]
        assert packet["list"]["version"] == "00"
        packet["list"]["frame_index"] = list_header[1]
        packet["list"]["length"] = list_header[2]
        if ( options.verbosity >= 1 ):
            print "list version:" , packet["list"]["version"]
            print "list frame_index:" , packet["list"]["frame_index"]
            print "list length:" , packet["list"]["length"]
        
        ack_message='{ "received" : "' + str(packet["list"]["frame_index"])
        frame_vote_max = -1
        bodypart_coords_est = {}
        for i in range(0, packet["list"]["length"]):
            data = conn.recv(unpacker_image_header.size)
            header_data = unpacker_image_header.unpack(data)
            image_header = {}
            image_header["version"] = header_data[0]
            assert image_header["version"] == "01"
            image_header["rows"] = header_data[1]
            image_header["cols"] = header_data[2]
            image_header["crop_x"] = header_data[3]
            image_header["crop_y"] = header_data[4]

            if ( options.verbosity >= 1 ):
                print "image header version:", image_header["version"]
                print "image header num rows:", image_header["rows"]
                print "image header num cols:", image_header["cols"]
                print "image header origin-x-coord:", image_header["crop_x"]
                print "image header origin-y-coord:", image_header["crop_y"]
        
            image_buffer_size = image_header["rows"]*image_header["cols"]
            data = bytearray(image_buffer_size)
            view = memoryview(data)
            toread = image_buffer_size
            while toread:
                nbytes = conn.recv_into(view, toread)
                view = view[nbytes:]
                toread -= nbytes
            if ( options.verbosity >= 1 ):
                print "Image Buffer Data: ", len(data)

            image_buffer = np.frombuffer(data, dtype = 'uint8')
        
            frame = image_buffer.reshape((image_header["rows"], image_header["cols"]))

            bodypart_vote_map = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 1), np.float)

            if (options.display_level >= 2):
                display_voters = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            ack_message += " " + str(image_header["rows"]) + "x" + str(image_header["cols"])
            kp_frame, desc_frame = surf.detectAndCompute(frame, None)
            # kp_frame = keypoints
            # desc_frame = descriptors

            for h, desc in enumerate(desc_frame):
                desc = np.array(desc, np.float32).reshape((1, 128))
                retval_pos, results_pos, neigh_resp_pos, dists_pos = bodypart_knn_pos.find_nearest(desc, 1)
                retval_neg, results_neg, neigh_resp_neg, dists_neg = bodypart_knn_neg.find_nearest(desc, 1)
                r_pos, d_pos = int(results_pos[0][0]), dists_pos[0][0]
                r_neg, d_neg = int(results_neg[0][0]), dists_neg[0][0]
                # r_pos, d_pos = bodypart_knn_pos.knnSearch(desc, 1, params = dict(checks = 8))
                # # print "Number of Positive Matches: ", d_pos
                # r_neg, d_neg = bodypart_knn_neg.knnSearch(desc, 1, params = dict(checks = 8))
                # print "Number of Negative Matches: ", d_neg
                relative_distance = d_pos - d_neg
                # print "Relative Distance: ", relative_distance

                if (relative_distance <= options.desc_distance_threshold):
                    # a = kp_frame[h].angle
                    a = np.pi*kp_frame[h].angle/180
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    p = kp_frame[h].pt + np.dot(R, bodypart_trained_data_pos.votes[r_pos])
                    x, y = p
                    # print "Vote Point: ", (x,y)
                    if (not (x <= options.vote_patch_size or x >= np.shape(frame)[1] - options.vote_patch_size or y <= options.vote_patch_size or y >= np.shape(frame)[0] - options.vote_patch_size)):
                        bodypart_vote_map[y - options.vote_patch_size:y + options.vote_patch_size + 1,
                                          x - options.vote_patch_size:x + options.vote_patch_size + 1] += bodypart_vote
                        if (options.display_level >= 2):
                            cv2.circle(display_voters, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)

            if (options.display_level >= 2):
                display_voters = cv2.resize(display_voters, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("voters", display_voters)

            vote_max = np.amax(bodypart_vote_map)
            print "Vote Max: ", vote_max
            if ( vote_max > options.vote_threshold and vote_max > frame_vote_max ):
                frame_vote_max = vote_max
                vote_max_loc = np.array(np.where(bodypart_vote_map == vote_max))
                vote_max_loc = vote_max_loc[:,0]
                bodypart_coords_est["conf"] = vote_max
                bodypart_coords_est["x"] = int(vote_max_loc[1]) + int(image_header["crop_x"])
                bodypart_coords_est["y"] = int(vote_max_loc[0]) + int(image_header["crop_y"])

            if (options.display_level >= 1):
                display_vote_map = np.array(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).copy(), np.float)
                display_vote_map /= 255.0
                bodypart_vote_map /= np.amax(bodypart_vote_map)
                display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
                if ( "x" in bodypart_coords_est ):
                    cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)

                # display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("voters", display_vote_map)
                cv2.waitKey(500)

        if ( "x" in bodypart_coords_est ):
            ack_message += '" , "detections" : [ { "frame_index" : ' + str(packet["list"]["frame_index"]) + ' , "test_bodypart" : "' + test_bodypart + '" , "coord_x" : ' + str(bodypart_coords_est["x"]) + ' , "coord_y" : ' + str(bodypart_coords_est["y"]) + ' , "conf" : ' + "{:.2f}".format(bodypart_coords_est["conf"]) + ' } ] }'
        else:
            ack_message += '" , "detections" : [ { "frame_index" : ' + str(packet["list"]["frame_index"]) + ' } ] }'

        if ( options.display_level >= 1 ):
            cv2.waitKey(100)
            cv2.destroyAllWindows()

        if ( options.verbosity >= 1 ):
            print "ack message:", ack_message
        header = ('01', len(ack_message))
        packed_ack_header = packer_ack_header.pack(*header)
        conn.sendall(packed_ack_header)
        conn.sendall(ack_message)

    conn.close()
    print 'Connection closed by', addr

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--positive-training-datafile", dest="train_data_p", help="File to save the information about the positive training data")
    parser.add_option("", "--negative-training-datafile", dest="train_data_n", help="File to save the information about the negative training data")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.1, help="threshold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15, help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=3.0, help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--vote-threshold", dest="vote_threshold", type="float", default=0.0, help="threshold on the net vote for a location for it to be a viable detection")
    parser.add_option("", "--display", dest="display_level", default=0, type="int", help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--nthread", dest="n_thread", type="int", default=1, help="maximum number of threads for multiprocessing")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--socket-port", dest="socket_port", type="int", default=9998, help="TCP port to listen and send to")
    parser.add_option("", "--verbosity", dest="verbosity", type="int", default=0, help="degree of verbosity")
    parser.add_option("", "--opencv", dest="opencv", type="int", default=0, help="degree of verbosity")

    (options, args) = parser.parse_args()

    main(options, args)

