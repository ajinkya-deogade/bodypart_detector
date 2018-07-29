#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle
import random
import re
import natsort

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/hcd uman_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
<<<<<<< HEAD
if __name__ == '__main__':
    test_annotation_file = '/c/Users/labadmin/Desktop/Rawdata_20140722_154715.mp4'
    frame_folder = 'C:\\Users\\labadmin\\Desktop\\Detection\\Rawdata_20140722_154715\\'
    test_annotation_file = '/Users/loaner/Desktop/Rawdata_20140722_154715.mp4'
    frame_folder = '/Users/loaner/Desktop/Detected/Rawdata_20140722_154715/'
    cap = cv2.VideoCapture(test_annotation_file)
    print cap
    resol_width = int(cap.get(3))
    resol_height = int(cap.get(4))
    fps = 45
    fourcc = int(cap.get(6))

    out = cv2.VideoWriter()
    print out
    success = out.open('C:\\Users\\labadmin\\Desktop\\Rawdata_20140722_154715_Detected.mp4',fourcc,fps,(1920,1920),True)
    fps = int(30)
    fourcc = int(cap.get(6))

    out = cv2.VideoWriter()
    success = out.open('~/Desktop/Rawdata_20140722_154715_Detected.mp4',fourcc,fps,(1920,1920),True)
    print success

    list_frames = os.listdir(frame_folder)
    list_frames = natsort.natsorted(list_frames, key=lambda y: y.lower())
    # print list_frames

    for frame in list_frames:
        frame_file = os.path.join(frame_folder, frame)
        frame_img = cv2.imread(frame_file)
        print frame
        # cv2.imshow('Frame',frame_img)
        # key_press = cv2.waitKey(-1)
        # if (key_press == 113 or key_press == 13):
        #     break
        out.write(frame_img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
=======

test_annotation_file = '/Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/4_20140214R.mp4'
frame_folder = '/Users/agomez/work/dev/mhdo/experiments/ad_7/expts/4_20140214R'
cap = cv2.VideoCapture(test_annotation_file)
resol_width = int(cap.get(3))
resol_height = int(cap.get(4))
fps = int(cap.get(5))
fourcc = int(cap.get(6))

out = cv2.VideoWriter()
success = out.open('~/Desktop/4_20140214R_Prediction.mp4',fourcc,fps,(1920,1920),True)
print success

list_frames = os.listdir(frame_folder)
list_frames = natsort.natsorted(list_frames, key=lambda y: y.lower())
# print list_frames

for frame in list_frames:
    frame_file = os.path.join(frame_folder, frame)
    frame_img = cv2.imread(frame_file)
    print frame
    # cv2.imshow('Frame',frame_img)
    # key_press = cv2.waitKey(-1)
    # if (key_press == 113 or key_press == 13):
    #     break
    out.write(frame_img)

cap.release()
out.release()
cv2.destroyAllWindows()
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7
