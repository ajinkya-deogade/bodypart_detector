#! /opt/local/bin/python

import json
import pickle
import re
import time
from optparse import OptionParser
from pyflann import *
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pylab
from sklearn import linear_model
import glob
import os
import csv

def stampAnnotations(video_file, metaDataFolderString):

    head, tail = os.path.split(video_file)
    trackerMetadataFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_metadata.csv'
    pythonMetadataFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_python.csv'

    print "Video File: ", video_file
    cap = cv2.VideoCapture(video_file)

    numberFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print 'Number of Video Frames Recorded : ', numberFrames

    trackerMetaData = np.empty((numberFrames, 61), dtype=np.float16)
    trackerMetaData[:] = np.NAN
    rowNum = -1
    with open(trackerMetadataFile, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            rowNum += 1
            colNum = -1
            if rowNum > 0:
                for val in row:
                    colNum += 1
                    try:
                        trackerMetaData[rowNum-1, colNum] = val
                    except:
                        print 'Number of Rows : ', rowNum
                        print 'Error in value ', val, rowNum-1, colNum
                        continue
    numberFramesTrackerRecorded = np.shape(trackerMetaData)[0]
    print 'Number of Metadata Frames Recorded : ', numberFramesTrackerRecorded

    pythonMetaData = np.empty((numberFrames, 61), dtype=np.float16)
    pythonMetaData[:] = np.NAN
    rowNum = -1
    with open(pythonMetadataFile, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            rowNum += 1
            colNum = -1
            if rowNum > 0:
                for val in row:
                    colNum += 1
                    try:
                        pythonMetaData[rowNum-1, colNum] = val
                    except:
                        print 'Number of Rows : ', rowNum
                        print 'Error in value ', val, rowNum-1, colNum
                        continue
    numberFramesPythonRecorded = np.shape(pythonMetaData)[0]
    print 'Number of Python Frames Recorded : ', numberFramesPythonRecorded

    f = 1
    crop_size = 512
    font = cv2.FONT_HERSHEY_SIMPLEX

    outputFrameSize = np.concatenate((np.multiply(np.ones((512, 512, 3), dtype=np.uint8), 255), np.multiply(np.ones((512, 100, 3), dtype=np.uint8), 255), np.multiply(np.ones((512, 512, 3), dtype=np.uint8), 255)), axis=1);
    height, width, layers = outputFrameSize.shape
    print height
    print width
    print layers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outputVideoFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_stitched.avi'
    video = cv2.VideoWriter(outputVideoFile, 0, 5, (width, height))

    for metaDataIndex in range(0, numberFramesTrackerRecorded-1):
        if cap.isOpened():
            f += 1
            try:
                ret, frame = cap.read()
            except:
                print 'Not able read frame %d' % (f)
                pass

            try:
                headX = trackerMetaData[metaDataIndex, 8]
                headY = trackerMetaData[metaDataIndex, 9]
                cropCenter_X = int(headX)
                cropCenter_Y = int(headY)

                fpgaFrameOverlay = frame.copy()
                fpgaFrame = frame.copy()
                fpgaLeftDorsalOrganLocation = {}
                fpgaRightDorsalOrganLocation = {}
                fpgaLeftDorsalOrganLocation['x'] = trackerMetaData[metaDataIndex, 18]
                fpgaLeftDorsalOrganLocation['y'] = trackerMetaData[metaDataIndex, 19]

                fpgaRightDorsalOrganLocation['x'] = trackerMetaData[metaDataIndex, 20]
                fpgaRightDorsalOrganLocation['y'] = trackerMetaData[metaDataIndex, 21]

                fpgaLeftDorsalOrganConfidence = trackerMetaData[metaDataIndex, 47]
                fpgaRightDorsalOrganConfidence = trackerMetaData[metaDataIndex, 48]
                if fpgaLeftDorsalOrganConfidence >= 7:
                    cv2.circle(fpgaFrameOverlay, (int(fpgaLeftDorsalOrganLocation['x']), int(fpgaLeftDorsalOrganLocation['y'])), 10, (255, 0, 255), thickness=-1)
                if fpgaRightDorsalOrganConfidence >= 7:
                    cv2.circle(fpgaFrameOverlay, (int(fpgaRightDorsalOrganLocation['x']), int(fpgaRightDorsalOrganLocation['y'])), 10, (0, 255, 255), thickness=-1)

                crop_x = max(0, cropCenter_X-int(crop_size/2))
                crop_y = max(0, cropCenter_Y-int(crop_size/2))
                fpgaFrameOverlay = fpgaFrameOverlay[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]
                fpgaFrame = fpgaFrame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]
                alpha = 0.4
                cv2.addWeighted(fpgaFrameOverlay, alpha, fpgaFrame, 1 - alpha, 0, fpgaFrame)

                cv2.putText(fpgaFrame, 'Left  DO: %05.2f' %(fpgaLeftDorsalOrganConfidence), (250, 40), font, 0.8, (255, 0, 255), 2)
                cv2.putText(fpgaFrame, 'Right DO: %05.2f' %(fpgaRightDorsalOrganConfidence), (250, 70), font, 0.8, (0, 255, 255), 2)

                pythonFrameOverlay = frame.copy()
                pythonFrame = frame.copy()
                pythonLeftDorsalOrganLocation = {}
                pythonRightDorsalOrganLocation = {}
                pythonLeftDorsalOrganLocation['x'] = pythonMetaData[metaDataIndex, 18]
                pythonLeftDorsalOrganLocation['y'] = pythonMetaData[metaDataIndex, 19]

                pythonRightDorsalOrganLocation['x'] = pythonMetaData[metaDataIndex, 20]
                pythonRightDorsalOrganLocation['y'] = pythonMetaData[metaDataIndex, 21]

                pythonLeftDorsalOrganConfidence = pythonMetaData[metaDataIndex, 47]
                pythonRightDorsalOrganConfidence = pythonMetaData[metaDataIndex, 48]

                if pythonLeftDorsalOrganConfidence >= 7:
                    cv2.circle(pythonFrameOverlay, (int(pythonLeftDorsalOrganLocation['x']), int(pythonLeftDorsalOrganLocation['y'])), 10, (255, 0, 255), thickness=-1)
                if pythonRightDorsalOrganConfidence >= 7:
                    cv2.circle(pythonFrameOverlay, (int(pythonRightDorsalOrganLocation['x']), int(pythonRightDorsalOrganLocation['y'])), 10, (0, 255, 255), thickness=-1)

                crop_x = max(0, cropCenter_X-int(crop_size/2))
                crop_y = max(0, cropCenter_Y-int(crop_size/2))
                pythonFrameOverlay = pythonFrameOverlay[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]
                pythonFrame = pythonFrame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]
                alpha = 0.4
                cv2.addWeighted(pythonFrameOverlay, alpha, pythonFrame, 1 - alpha, 0, pythonFrame)

                cv2.putText(pythonFrame, 'Left  DO: %05.2f' %(pythonLeftDorsalOrganConfidence), (250, 40), font, 0.8, (255, 0, 255), 2)
                cv2.putText(pythonFrame, 'Right DO: %05.2f' %(pythonRightDorsalOrganConfidence), (250, 70), font, 0.8, (0, 255, 255), 2)

                # cv2.imshow('python', pythonFrame)
                # cv2.waitKey(2000)
                pythonFPGAFrame = np.concatenate((pythonFrame, np.multiply(np.ones((512, 100, 3), dtype=np.uint8), 255), fpgaFrame), axis=1)
                video.write(pythonFPGAFrame)

                # cv2.imshow('Python FPGA', pythonFPGAFrame)
                # cv2.waitKey(2000)
            except:
                continue

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--video-file", dest="video_file", default="", help="video file to be annotated")
    parser.add_option("", "--meta-data-folder", dest="metaDataFolderString", default="", help="video file to be annotated")

    (options, args) = parser.parse_args()

    stampAnnotations(options.video_file, options.metaDataFolderString)