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

def stampAnnotations(video_file, project_dir, metaDataFolderString, plot_body_part):

    video_file = re.sub(".*/data/", "data/", video_file)
    video_file = os.path.join(project_dir, video_file)

    head, tail = os.path.split(video_file)
    trackerMetadataFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_metadata.csv'

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

    f = 0
    crop_size = 512
    spotSize = 20
    confThreshold = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    outputFrameSize = np.multiply(np.ones((512, 512, 3), dtype=np.uint8), 255)
    height, width, layers = outputFrameSize.shape
    print "Video Size : %d x %d x %d" %(height, width, layers)
    print "Spot Size : ", spotSize

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outputVideoFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_spotSize_%d_%s.avi' %(spotSize, plot_body_part)
    video = cv2.VideoWriter(outputVideoFile, 0, 5, (width, height))

    frameNumber = 0
    for metaDataIndex in range(0, numberFramesTrackerRecorded-1):
        actualIndex = int(trackerMetaData[metaDataIndex, 0]) - int(trackerMetaData[0, 0])
        # print 'Actual Index : ', actualIndex
        headX = trackerMetaData[metaDataIndex, 8]
        headY = trackerMetaData[metaDataIndex, 9]
        cropCenter_X = int(headX)
        cropCenter_Y = int(headY)

        if plot_body_part == 'LeftMHhook':
            xIndex, yIndex, confIndex = 14, 15, 45
        elif plot_body_part == 'RightMHhook':
            xIndex, yIndex, confIndex = 16, 17, 46
        elif plot_body_part == 'LeftDorsalOrgan':
            xIndex, yIndex, confIndex = 18, 19, 47
        elif plot_body_part == 'RightDorsalOrgan':
            xIndex, yIndex, confIndex = 20, 21, 48
        else:
            xIndex, yIndex, confIndex = 12, 13, 44

        fpgaBodyPartInfo = {}
        fpgaBodyPartInfo['x'] = trackerMetaData[metaDataIndex, xIndex]
        fpgaBodyPartInfo['y'] = trackerMetaData[metaDataIndex, yIndex]
        fpgaBodyPartInfo['conf'] = trackerMetaData[metaDataIndex, confIndex]

        fpgaMouthHookInfo = {}
        fpgaMouthHookInfo['x'] = trackerMetaData[metaDataIndex, 12]
        fpgaMouthHookInfo['y'] = trackerMetaData[metaDataIndex, 13]
        fpgaMouthHookInfo['conf'] = trackerMetaData[metaDataIndex, 44]

        if cap.isOpened():
            try:
                cap.set(1, actualIndex - 1)
                ret, originalFrame = cap.read()
                frameNumber += 1
            except:
                print 'Not able read frame %d' % (f)
                continue

            frame = originalFrame.copy()
            frameOverlay = originalFrame.copy()

            if fpgaBodyPartInfo['conf'] >= confThreshold:
                recTopLeft = {}
                recTopLeft['x'] = int(fpgaBodyPartInfo['x']) - int(spotSize/2)
                recTopLeft['y'] = int(fpgaBodyPartInfo['y']) - int(spotSize/2)

                recBottomRight = {}
                recBottomRight['x'] = int(fpgaBodyPartInfo['x']) + int(spotSize/2)
                recBottomRight['y'] = int(fpgaBodyPartInfo['y']) + int(spotSize/2)
                cv2.rectangle(frameOverlay, (recTopLeft['x'], recTopLeft['y']), (recBottomRight['x'], recBottomRight['y']), (0, 255, 0), thickness=-1)

            crop_x = max(0, cropCenter_X-int(crop_size/2))
            crop_y = max(0, cropCenter_Y-int(crop_size/2))
            frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]
            frameOverlay = frameOverlay[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]
            alpha = 0.3
            cv2.addWeighted(frameOverlay, alpha, frame, 1 - alpha, 0, frame)

            cv2.putText(frame, 'Spot Size %d pxl' %(spotSize), (350, 40), font, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, 'Confidence %d ' % (fpgaBodyPartInfo['conf']), (350, 60), font, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, '%s/%s' %(str(frameNumber), str(metaDataIndex+1)), (40, 40), font, 0.5, (255, 255, 255), 2)

            # video.write(frame)
            cv2.imshow('Python FPGA Annotation', frame)
            cv2.waitKey(1000)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--video-file-list", dest="video_file_list", default="", help="video file to be annotated")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--plot-body-part", dest="body_part", default="", help="video file to be annotated")
    parser.add_option("", "--meta-data-folder", dest="metaDataFolderString", default="", help="video file to be annotated")

    (options, args) = parser.parse_args()

    tempFile = []
    with open(options.video_file_list) as all_list:
        for vidFile in all_list:
            tempFile.append(vidFile)
    video_file = tempFile[0]

    stampAnnotations(video_file, options.project_dir, options.metaDataFolderString, options.body_part)