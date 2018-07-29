#! /usr/bin/env python

from optparse import OptionParser
import cv2
import re
import copy
import numpy as np

r_coords = re.compile(r'\((?P<x>[^,]+),(?P<y>[^,]+),(?P<w>[^,]+),(?P<h>.+)\)')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--file", dest="video_file", default="", help="video file to play")
    parser.add_option("", "--bbox", dest="bounding_box", default="", help="bounding box to be tracked (x,y,w,h)")

    (options, args) = parser.parse_args()

    print "video_file:" , options.video_file

    coords_match = r_coords.match(options.bounding_box)

    bbox_init = {}
    if (coords_match != None):
        bbox_init["x1"] = int(coords_match.group("x"))
        bbox_init["y1"] = int(coords_match.group("y"))
        bbox_init["w"] = int(coords_match.group("w"))
        bbox_init["h"] = int(coords_match.group("h"))
        bbox_init["x2"] = int(coords_match.group("w")) + bbox_init["x1"]
        bbox_init["y2"] = int(coords_match.group("h")) + bbox_init["y1"]
    else:
        print "Error parsing bounding box from input argument"
        exit

    video_in = cv2.VideoCapture(options.video_file)
    fps = video_in.get(cv2.cv.CV_CAP_PROP_FPS)
    print "fps:" , fps

    frame_dur = int(1000.0 / float(fps))
    print "frame duration (ms):" , frame_dur

    bbox_prev = copy.deepcopy(bbox_init)
    bbox = copy.deepcopy(bbox_init)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    surf = cv2.SURF(300, nOctaves=4, nOctaveLayers=3)

    frame_id = 0
    kp_template = []
    desc_template = []
    while (True):
        ret, frame_orig = video_in.read()

        if (ret == False):
            break

        frame = cv2.cvtColor(frame_orig, cv2.cv.CV_RGB2GRAY)
#        frame = clahe.apply(frame)

        if (frame_id <= 5):
            kp, desc = surf.detectAndCompute(frame[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]], None)
            for k in kp:
                x,y = k.pt
                x += int(bbox["x1"])
                y += int(bbox["y1"])
                k.pt = (x, y)

            kp_template.extend(kp)
            desc_template.extend(desc)

            display_template = cv2.drawKeypoints( frame[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]], kp, None, (0, 255,0), 4)
            cv2.imshow("Template", display_template)
            
        display_frame = frame_orig.copy()

        if (frame_id >= 6):
            kp_frame, desc_frame = surf.detectAndCompute(frame, None)
            knn = cv2.KNearest()
            desc_frame_samples = np.array(desc_frame)
            kp_frame_responses = np.arange(len(kp_frame),dtype = np.float32)
            knn.train(desc_frame_samples,kp_frame_responses)
            
            dx_votes = []
            dy_votes = []
            for h, desc in enumerate(desc_template):
                desc = np.array(desc,np.float32).reshape((1,128))
                retval, results, neigh_resp, dists = knn.find_nearest(desc,1)
                r,d =  int(results[0][0]),dists[0][0]
                
                x, y = kp_frame[r].pt
                # x = int(x + bbox["x1"] - 50)
                # y = int(y + bbox["y1"] - 50)

                if (d < 0.05):
                    color = (0, 0, 255)
                    radius = 20
                    thickness = 3

                    xt, yt = kp_template[h].pt
                    dx_votes.append(x)
                    dy_votes.append(y)

                else:
                    color = (100, 0, 0)
                    radius = 10
                    thickness = 2
                    
                center = (int(x), int(y))
                cv2.circle(display_frame, center, radius, color, thickness)

            dx = int(np.median(dx_votes))
            dy = int(np.median(dy_votes))
            print dx, dy

            bbox["x1"] = int(dx - bbox["w"]/2)
            bbox["y1"] = int(dy - bbox["h"]/2)
            bbox["x2"] = int(dx + bbox["w"]/2)
            bbox["y2"] = int(dy + bbox["h"]/2)

            print "bbox: " , bbox
            cv2.rectangle(display_frame, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), (0, 255, 0), 2)

        display_frame = cv2.resize(display_frame, (0,0), fx=0.25, fy=0.25)
        cv2.imshow(options.video_file, display_frame)

        if (frame_id == 0):
            k = cv2.waitKey(-1)
        else:
            k = cv2.waitKey(3)

        if (k == 'q'):
            exit

        frame_id = frame_id + 1

    video_in.release()
