#! /usr/bin/env python

from optparse import OptionParser
import json
import os
import struct
import cv2
import numpy as np
import socket
import sys
import copy
from scipy import interpolate


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
    np.seterr(divide='ignore', invalid='ignore')
    # bodypart_coords = {}
    bodypart_coords_new = {}
    Distance_Interpolated_Estimated_All = []
    bodypart_coords_new["TotalLength_Not_Consecutive_Detections"] = []
    bodypart_coords_new["TotalLength_Consecutive_Detections"] = []

    for videonumber in range(0, n_server):
        socks.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        server_free.append(True)

    test_bodypart = options.test_bodypart

    with open(options.start_end_file) as fin_StartEnd:
        bodypart_coords = dict(json.load(fin_StartEnd))
        # print np.shape(bodypart_coords)
        # pprint(bodypart_coords[str(0)])

    # pprint(bodypart_coords)

    frame_index = -1

    newCoodinateFile = os.path.join('../expts/new_coordinates_' + test_bodypart.strip() + ".json")
    fileWriter_newCoordinates = open(newCoodinateFile, 'w+')

    for videonumber in range(0, len(bodypart_coords)):

        video_file = bodypart_coords[str(videonumber)]["VideoFile"]
        print "Video File: ", video_file
        cap = cv2.VideoCapture(video_file)

        bodypart_coords_new[videonumber] = {}
        bodypart_coords_new[videonumber]["VideoFile"] = video_file
        bodypart_coords_new[videonumber]["Coordinates"] = {}
        bodypart_coords_new[videonumber]["Frame_Sequence"] = {}

        n_start = len(bodypart_coords[str(videonumber)]["StartFrames"])
        n_end = len(bodypart_coords[str(videonumber)]["EndFrames"])
        n_bodypart_visible_seq = 0

        if n_start == n_end:
            n_bodypart_visible_seq = n_start
            print "Number of Start and End points for video: ",n_bodypart_visible_seq
        else:
            print "Number of Start and End points not equal for video: ", str(videonumber),";  ", video_file
            continue

        if (cap.isOpened()):
            for n_seq in range(0, n_bodypart_visible_seq):
                brk_seq = 0
                cont_frames = []
                discont_frames = []
                consecutive_frames = []
                not_consecutive_frames = []
                n_consecutive_frames = []
                n_not_consecutive_frames = []
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq] = {}
                for i in range(0, len(bodypart_coords[str(videonumber)])-3):
                    # print bodypart_coords[str(videonumber)][str(i)]["FrameIndexVideo"]
                    # print "Type EndFrame: ",type(bodypart_coords[str(videonumber)]["EndFrames"][n_seq])
                    # print "Type FrameIndexVideo: ",type(bodypart_coords[str(videonumber)][str(i)]["FrameIndexVideo"])
                    # print bodypart_coords[str(videonumber)]["StartFrames"][n_seq]
                    if int(bodypart_coords[str(videonumber)][str(i)]["FrameIndexVideo"]) == int(bodypart_coords[str(videonumber)]["StartFrames"][n_seq]):
                        i_start = i
                    if int(bodypart_coords[str(videonumber)][str(i)]["FrameIndexVideo"]) == int(bodypart_coords[str(videonumber)]["EndFrames"][n_seq]):
                        i_end = i
                seqLength_index =  i_end - i_start

                # print "Sequence Length: ", seqLength_index
                # print "Start X : ", int(bodypart_coords[str(videonumber)][str(i_start)]["x"])
                # print "END X: ", int(bodypart_coords[str(videonumber)][str(i_end)]["x"])
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["StartFrame"] = bodypart_coords[str(videonumber)][str(i_start)]["FrameIndexVideo"]
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["EndFrame"] = bodypart_coords[str(videonumber)][str(i_end)]["FrameIndexVideo"]

                for n in range(i_start, i_end):
                    annotation_interval = bodypart_coords[str(videonumber)][str(n+1)]["FrameIndexVideo"] - bodypart_coords[str(videonumber)][str(n)]["FrameIndexVideo"]
                    # print "Sequence %d - Subsequence Index: %d" %(n_seq, n)
                    x_endpoints = [bodypart_coords[str(videonumber)][str(n)]["x"], bodypart_coords[str(videonumber)][str(n+1)]["x"]]
                    y_endpoints = [bodypart_coords[str(videonumber)][str(n)]["y"], bodypart_coords[str(videonumber)][str(n+1)]["y"]]
                    # print "X-Endpoints", x_endpoints
                    # print "Y-Endpoints", y_endpoints
                    interp_func = interpolate.interp1d(x_endpoints, y_endpoints)

                    if x_endpoints[0] == x_endpoints[1]:
                        x_interp = np.linspace(x_endpoints[0], x_endpoints[1], annotation_interval+1)
                        x_interp = [int(x) for x in x_interp]
                        y_interp = np.linspace(y_endpoints[0], y_endpoints[1], annotation_interval+1)
                        y_interp = [int(y) for y in y_interp]

                    else:
                        # print "Interpolated.........."
                        x_interp = np.linspace(x_endpoints[0], x_endpoints[1], annotation_interval+1)
                        x_interp = [int(x) for x in x_interp]
                        y_interp = interp_func(x_interp)
                        y_interp = [int(y) for y in y_interp]

                    # print "X-Values: ",x_interp
                    # print "Y-Values: ",y_interp
                    local_iter = -1

                    for frameIndex in range(int(bodypart_coords[str(videonumber)][str(n)]["FrameIndexVideo"]), int(bodypart_coords[str(videonumber)][str(n+1)]["FrameIndexVideo"])):
                        cap.set(1, float(frameIndex))
                        ret,frame = cap.read()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex] = {}
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"] = {}
                        # bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["x"] = -1
                        # bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["y"] = -1
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"] = {}
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"] = {}
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["FrameIndexVideo"] = frameIndex
                        if (not ret):
                            continue
                        # cv2.imshow("Frames",frame)
                        # cv2.waitKey(1000)
                        local_iter += 1
                        frame_index += 1
                        images = []
                        packed_image_headers = []
                        images_data = []
                        # start_time = time.clock()
                        image = copy.deepcopy(frame)

                        crop_x = 0
                        crop_y = 0
                        # image = image[crop_y:1920,crop_x:1920,0]

                        images.append(copy.deepcopy(image))
                        image_info = np.shape(image)
                        # print image_info
                        if not image_info:
                            print "Empty Image Video %d, Frame %d" % (str(videonumber), frameIndex)
                            continue

                        print "Frame Index: ", frame_index
                        # print "Local Iter: ", local_iter

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
                                                bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"]["x"] = \
                                                    received_json["detections"][di]["coord_x"]
                                                bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"]["y"] = \
                                                    received_json["detections"][di]["coord_y"]
                                                bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"]["Confidence"] = received_json["detections"][di]["conf"]
                                                # print "Received Confidence:", bodypart_coords_est[str(videonumber)][frameNumber]["conf"]
                                            else:
                                                bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"]["x"] = -1
                                                bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"]["x"] = -1
                                except socket.timeout:
                                    pass
                                except:
                                    print "Unexpected error:", sys.exc_info()[0]

                        if frameIndex == bodypart_coords[str(videonumber)][str(n)]["FrameIndexVideo"]:
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["x"] = bodypart_coords[str(videonumber)][str(n)]["x"]
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["y"] = bodypart_coords[str(videonumber)][str(n)]["y"]
                        elif frameIndex == bodypart_coords[str(videonumber)][str(n+1)]["FrameIndexVideo"]:
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["x"] = bodypart_coords[str(videonumber)][str(n+1)]["x"]
                            bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Annotated"]["y"] = bodypart_coords[str(videonumber)][str(n+1)]["y"]

                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"]["x"] = x_interp[local_iter]
                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"]["y"] = y_interp[local_iter]

                        bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Distance_Interpolated_Estimated"] = np.sqrt(np.square(bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"]["x"] - bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"]["x"]) + np.square(bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Interpolated"]["y"] - bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Estimated"]["y"]))
                        if bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Distance_Interpolated_Estimated"] < options.outlier_error_dist:
                            cont_frames.append(frameIndex)
                            if discont_frames:
                                not_consecutive_frames.append(discont_frames)
                                n_not_consecutive_frames.append(len(discont_frames))
                            discont_frames = []
                        else:
                            discont_frames.append(frameIndex)
                            # brk_seq += 1
                            if cont_frames:
                                consecutive_frames.append(cont_frames)
                                n_consecutive_frames.append(len(cont_frames))
                            cont_frames = []

                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["Length_Sequence"] = float(float(seqLength_index)*float(annotation_interval))
                # bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["Number of Breaks (>10)"] = brk_seq
                # perc = float((float(brk_seq)/float(float(seqLength_index)*float(annotation_interval)))*float(100))
                # bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["Percentage of Breaks (>10)"] = perc
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["Consecutive_Sequence_Indices"] = consecutive_frames
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["Not_Consecutive_Sequence_Indices"] = not_consecutive_frames
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["Length_Consecutive_Detections"] = n_consecutive_frames
                bodypart_coords_new[videonumber]["Frame_Sequence"][n_seq]["Length_Not_Consecutive_Detections"] = n_not_consecutive_frames
                bodypart_coords_new["TotalLength_Not_Consecutive_Detections"].extend(n_not_consecutive_frames)
                bodypart_coords_new["TotalLength_Consecutive_Detections"].extend(n_consecutive_frames)

        Distance_Interpolated_Estimated = []
        for frameIndex in bodypart_coords_new[videonumber]["Coordinates"]:
            Distance_Interpolated_Estimated.append(bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Distance_Interpolated_Estimated"])
            Distance_Interpolated_Estimated_All.append(bodypart_coords_new[videonumber]["Coordinates"][frameIndex]["Distance_Interpolated_Estimated"])

        bodypart_coords_new[videonumber]["Median_Distance_Interpolated_Estimated"] = np.median(Distance_Interpolated_Estimated)
        bodypart_coords_new[videonumber]["Mean_Distance_Interpolated_Estimated"] = np.mean(Distance_Interpolated_Estimated)

    bodypart_coords_new["Median_Distance_Interpolated_Estimated"] = np.median(Distance_Interpolated_Estimated_All)
    bodypart_coords_new["Mean_Distance_Interpolated_Estimated"] = np.mean(Distance_Interpolated_Estimated_All)
    bodypart_coords_new["Mean_Length_Consecutive_Detections"] = np.mean(bodypart_coords_new["TotalLength_Consecutive_Detections"])
    bodypart_coords_new["Median_Length_Consecutive_Detections"] = np.median(bodypart_coords_new["TotalLength_Consecutive_Detections"])
    bodypart_coords_new["Mean_Length_Not_Consecutive_Detections"] = np.mean(bodypart_coords_new["TotalLength_Not_Consecutive_Detections"])
    bodypart_coords_new["Median_Length_Not_Consecutive_Detections"] = np.median(bodypart_coords_new["TotalLength_Not_Consecutive_Detections"])
    bodypart_coords_new["BodyPart"] = options.test_bodypart
    bodypart_coords_new["OutlierErrorDistance"] = options.outlier_error_dist
    json.dump(bodypart_coords_new, fileWriter_newCoordinates, sort_keys=True, indent=4, separators=(',', ': '))
    fileWriter_newCoordinates.close()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--start-end-file", dest="start_end_file", default="", help="path containing data directory")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,
                      help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",
                      help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--test-bodypart", dest="test_bodypart", default="MouthHook",
                      help="Input the bodypart to be tested")

    (options, args) = parser.parse_args()

    main(options, args)

