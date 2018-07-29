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
import SocketServer

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

surf = cv2.SURF(600, nOctaves=3, nOctaveLayers=3)

test_bodypart = None
bodypart_knn_pos = cv2.KNearest()
bodypart_knn_neg = cv2.KNearest()
bodypart_trained_data_pos = None
bodypart_vote = []

class MyTCPHandler(SocketServer.StreamRequestHandler):

    def handle(self):
        # self.rfile is a file-like object created by the handler;
        # we can now use e.g. readline() instead of raw recv() calls
        self.request.setblocking(1)

        self.data = self.request.recv(unpacker_header.size)
        self.header = unpacker_header.unpack(self.data)
        print "{} wrote:".format(self.client_address[0])
        packet = {}
        packet["version"] = self.header[0]
        assert packet["version"] == "01"
        packet["size"] = self.header[1]
        print "packet version:" , packet["version"]
        print "packet size:" , packet["size"]

        self.data = self.request.recv(unpacker_list_header.size)
        self.list_header = unpacker_list_header.unpack(self.data)
        packet["list"] = {}
        packet["list"]["version"] = self.list_header[0]
        assert packet["list"]["version"] == "00"
        packet["list"]["frame_index"] = self.list_header[1]
        packet["list"]["length"] = self.list_header[2]
        print "list version:" , packet["list"]["version"]
        print "list frame_index:" , packet["list"]["frame_index"]
        print "list length:" , packet["list"]["length"]
        
        ack_message='{ "received" : "' + str(packet["list"]["frame_index"])
        frame_vote_max = -1
        bodypart_coords_est = {}
        for i in range(0, packet["list"]["length"]):
            self.data = self.request.recv(unpacker_image_header.size)
            self.image_header = unpacker_image_header.unpack(self.data)
            image_header = {}
            image_header["version"] = self.image_header[0]
            assert image_header["version"] == "01"
            image_header["rows"] = self.image_header[1]
            image_header["cols"] = self.image_header[2]
            image_header["crop_x"] = self.image_header[3]
            image_header["crop_y"] = self.image_header[4]

            print "image header version:", image_header["version"]
            print "image header num rows:", image_header["rows"]
            print "image header num cols:", image_header["cols"]
            print "image header origin-x-coord:", image_header["crop_x"]
            print "image header origin-y-coord:", image_header["crop_y"]
        
            image_buffer_size = image_header["rows"]*image_header["cols"]
            self.data = bytearray(image_buffer_size)
            view = memoryview(self.data)
            toread = image_buffer_size
            while toread:
                nbytes = self.request.recv_into(view, toread)
                view = view[nbytes:]
                toread -= nbytes
            print len(self.data)
            self.image_buffer = np.frombuffer(self.data, dtype = 'uint8')
        
            frame = self.image_buffer.reshape((image_header["rows"], image_header["cols"]))

            bodypart_vote_map = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 1), np.float)
            if (options.display_level >= 2):
                display_voters = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            kp_frame, desc_frame = surf.detectAndCompute(frame, None)
            for h, desc in enumerate(desc_frame):
                desc = np.array(desc, np.float32).reshape((1, 128))
                retval_pos, results_pos, neigh_resp_pos, dists_pos = bodypart_knn_pos.find_nearest(desc, 1)
                retval_neg, results_neg, neigh_resp_neg, dists_neg = bodypart_knn_neg.find_nearest(desc, 1)
                r_pos, d_pos = int(results_pos[0][0]), dists_pos[0][0]
                r_neg, d_neg = int(results_neg[0][0]), dists_neg[0][0]
                relative_distance = d_pos - d_neg

                if (relative_distance <= options.desc_distance_threshold):
                    a = np.pi * kp_frame[h].angle / 180.0
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    p = kp_frame[h].pt + np.dot(R, bodypart_trained_data_pos.votes[r_pos])
                    x, y = p
                    if (not (x <= options.vote_patch_size or x >= np.shape(frame)[1] - options.vote_patch_size or y <= options.vote_patch_size or y >= np.shape(frame)[0] - options.vote_patch_size)):
                        bodypart_vote_map[y - options.vote_patch_size:y + options.vote_patch_size + 1,
                                          x - options.vote_patch_size:x + options.vote_patch_size + 1] += bodypart_vote
                        if (options.display_level >= 2):
                            cv2.circle(display_voters, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)

            if (options.display_level >= 2):
                display_voters = cv2.resize(display_voters, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("voters", display_voters)

            vote_max = np.amax(bodypart_vote_map)
            if ( vote_max > 0 and vote_max > frame_vote_max ):
                frame_vote_max = vote_max
                vote_max_loc = np.array(np.where(bodypart_vote_map == vote_max))
                vote_max_loc = vote_max_loc[:,0]
                bodypart_coords_est["conf"] = vote_max
                bodypart_coords_est["x"] = int(vote_max_loc[1]) + int(image_header["crop_x"])
                bodypart_coords_est["y"] = int(vote_max_loc[0]) + int(image_header["crop_y"])
            else:
                bodypart_coords_est = None

            if (options.display_level >= 1):
                display_vote_map = np.array(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR).copy(), np.float)
                display_vote_map /= 255.0
                bodypart_vote_map /= np.amax(bodypart_vote_map)
                display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
                if ( bodypart_coords_est != None ):
                    cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
                display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("voters", display_vote_map)

            ack_message += " " + str(image_header["rows"]) + "x" + str(image_header["cols"])
        
        if ( "x" in bodypart_coords_est ):
            ack_message += '" , "detections" : [ { "frame_index" : ' + str(packet["list"]["frame_index"]) + ' , "test_bodypart" : "' + test_bodypart + '" , "coord_x" : ' + str(bodypart_coords_est["x"]) + ' , "coord_y" : ' + str(bodypart_coords_est["y"]) + ' , "conf" : ' + "{:.2f}".format(bodypart_coords_est["conf"]) + ' } ] }'
        else:
            ack_message += '" , "detections" : [ { "frame_index" : ' + str(packet["list"]["frame_index"]) + ' } ] }'

        if ( options.display_level >= 1):
            cv2.waitKey(100)
            cv2.destroyAllWindows()

        # Likewise, self.wfile is a file-like object used to write back
        # to the client
        print "ack message:", ack_message
        header = ('01', len(ack_message))
        packed_ack_header = packer_ack_header.pack(*header)
        self.request.sendall(packed_ack_header)
        self.request.sendall(ack_message)


def main(options, args):
    global test_bodypart
    global bodypart_knn_pos, bodypart_knn_neg, bodypart_trained_data_pos, bodypart_vote

    bodypart_trained_data_pos = SaveClass()
    bodypart_trained_data_pos = pickle.load(open(options.train_data_p, 'rb'))
    bodypart_trained_data_neg = SaveClass()
    bodypart_trained_data_neg = pickle.load(open(options.train_data_n, 'rb'))

    test_bodypart = bodypart_trained_data_neg.bodypart
    print "test_bodypart:" , test_bodypart

    bodypart_knn_pos.train(bodypart_trained_data_pos.descriptors, bodypart_trained_data_pos.keypoints)
    bodypart_knn_neg.train(bodypart_trained_data_neg.descriptors, bodypart_trained_data_neg.keypoints)

    bodypart_vote = np.zeros((2 * options.vote_patch_size + 1, 2 * options.vote_patch_size + 1, 1), np.float)
    
    for x in range(-options.vote_patch_size, options.vote_patch_size + 1):
        for y in range(-options.vote_patch_size, options.vote_patch_size + 1):
            bodypart_vote[y + options.vote_patch_size, x + options.vote_patch_size] = 1.0 + np.exp(
                -0.5 * (x * x + y * y) / (np.square(options.vote_sigma))) / (options.vote_sigma * np.sqrt(2 * np.pi))

    HOST, PORT = "localhost", options.socket_port

    # Create the server, binding to localhost on port 9999
    SocketServer.TCPServer.allow_reuse_address = True
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--positive-training-datafile", dest="train_data_p", help="File to save the information about the positive training data")
    parser.add_option("", "--negative-training-datafile", dest="train_data_n", help="File to save the information about the negative training data")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.1,help="threshold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15,help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=3.0,help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--nthread", dest="n_thread", type="int", default=1, help="maximum number of threads for multiprocessing")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--socket-port", dest="socket_port", type="int", default=9998, help="TCP port to listen and send to")

    (options, args) = parser.parse_args()

    main(options, args)

