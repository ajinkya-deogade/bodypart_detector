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

def stampAnnotations(video_file, metaDataFolderString, plot_body_part):

    head, tail = os.path.split(video_file)
    trackerMetadataFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_metadata.csv'
    pythonMetadataFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_python.csv'
    annotationMetadataFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_annotation.csv'

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

    annotationMetaData = np.empty((numberFrames, 61), dtype=np.float16)
    annotationMetaData[:] = np.NAN
    rowNum = -1
    with open(annotationMetadataFile, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            rowNum += 1
            colNum = -1
            if rowNum > 0:
                for val in row:
                    colNum += 1
                    try:
                        annotationMetaData[rowNum-1, colNum] = val
                    except:
                        print 'Number of Rows : ', rowNum
                        print 'Error in value ', val, rowNum-1, colNum
                        continue
    numberFramesAnnotationRecorded = np.shape(annotationMetaData)[0]
    print 'Number of Annotation Frames Recorded : ', numberFramesAnnotationRecorded


    f = 0
    crop_size = 512
    font = cv2.FONT_HERSHEY_SIMPLEX

    outputFrameSize = np.multiply(np.ones((512, 512, 3), dtype=np.uint8), 255)
    height, width, layers = outputFrameSize.shape
    print "Video Size : %d x %d x %d" %(height, width, layers)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # outputVideoFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_fpga_python_annotation_LDO.avi'
    outputVideoFile = os.path.join(head, metaDataFolderString, os.path.splitext(tail)[0]) + '_fpga_python_annotation_%s.avi' %(plot_body_part)
    video = cv2.VideoWriter(outputVideoFile, 0, 5, (width, height))

    numberFramesAnnotationRecorded = 501
    frameNumber = 0
    for metaDataIndex in range(0, numberFramesAnnotationRecorded-1):
        actualIndex = int(trackerMetaData[metaDataIndex, 0]) - int(trackerMetaData[0, 0])
        print 'Actual Index : ', actualIndex
        headX = trackerMetaData[metaDataIndex, 8]
        headY = trackerMetaData[metaDataIndex, 9]
        cropCenter_X = int(headX)
        cropCenter_Y = int(headY)

        if plot_body_part == 'LeftDorsalOrgan':
            xIndex, yIndex = 18, 19
        elif plot_body_part == 'RighrDorsalOrgan':
            xIndex, yIndex = 20, 21
        elif plot_body_part == 'RightMHhook':
            xIndex, yIndex = 16, 17
        elif plot_body_part == 'LeftMHhook':
            xIndex, yIndex = 14, 15
        else:
            xIndex, yIndex = 12, 13

        fpgaBodyPartLocation = {}
        fpgaBodyPartLocation['x'] = trackerMetaData[metaDataIndex, xIndex]
        fpgaBodyPartLocation['y'] = trackerMetaData[metaDataIndex, yIndex]

        annotationBodyPartOrganLocation = {}
        annotationBodyPartOrganLocation['x'] = annotationMetaData[metaDataIndex, xIndex]
        annotationBodyPartOrganLocation['y'] = annotationMetaData[metaDataIndex, yIndex]
        print annotationBodyPartOrganLocation

        pythonBodyPartLocation = {}
        pythonBodyPartLocation['x'] = pythonMetaData[metaDataIndex, xIndex]
        pythonBodyPartLocation['y'] = pythonMetaData[metaDataIndex, yIndex]

        fpgaMouthHookLocation = {}
        fpgaMouthHookLocation['x'] = trackerMetaData[metaDataIndex, 12]
        fpgaMouthHookLocation['y'] = trackerMetaData[metaDataIndex, 13]

        annotationMouthHookLocation = {}
        annotationMouthHookLocation['x'] = annotationMetaData[metaDataIndex, 12]
        annotationMouthHookLocation['y'] = annotationMetaData[metaDataIndex, 13]

        pythonMouthHookLocation = {}
        pythonMouthHookLocation['x'] = pythonMetaData[metaDataIndex, 12]
        pythonMouthHookLocation['y'] = pythonMetaData[metaDataIndex, 13]

        # if annotationLeftDorsalOrganLocation['x'] != -1 and annotationLeftDorsalOrganLocation['y'] != -1:
        if annotationMouthHookLocation['x'] != -1 and annotationMouthHookLocation['y'] != -1:
            if cap.isOpened():
                try:
                    cap.set(1, actualIndex - 1)
                    ret, originalFrame = cap.read()
                    print np.shape(originalFrame)
                    f += 1
                except:
                    print 'Not able read frame %d' % (f)
                    pass

                frameNumber += 1
                frame = originalFrame.copy()

                ## Other Body Part
                cv2.putText(frame, '*', (int(fpgaBodyPartLocation['x']), int(fpgaBodyPartLocation['y'])), font, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, '*', (int(pythonBodyPartLocation['x']), int(pythonBodyPartLocation['y'])), font, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, '*', (int(annotationBodyPartOrganLocation['x']), int(annotationBodyPartOrganLocation['y'])), font, 0.6, (255, 255, 255), 2)

                ## Mouth Hook
                cv2.putText(frame, '+', (int(fpgaMouthHookLocation['x']), int(fpgaMouthHookLocation['y'])), font, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, '+', (int(pythonMouthHookLocation['x']), int(pythonMouthHookLocation['y'])), font, 0.8, (255, 0, 0), 2)
                cv2.putText(frame, '+', (int(annotationMouthHookLocation['x']), int(annotationMouthHookLocation['y'])), font, 0.8, (255, 255, 255), 2)

                crop_x = max(0, cropCenter_X-int(crop_size/2))
                crop_y = max(0, cropCenter_Y-int(crop_size/2))
                frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]

                cv2.putText(frame, 'FPGA', (400, 40), font, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, 'PYTHON', (400, 60), font, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, 'ANNOTATION', (400, 80), font, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, '*  %s' % (plot_body_part), (200, 40), font, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, '+  Mouth Hook', (200, 70), font, 0.6, (255, 255, 255), 2)

                cv2.putText(frame, str(frameNumber), (40, 40), font, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, str(metaDataIndex+1), (40, 65), font, 0.6, (50, 50, 50), 2)

                video.write(frame)
                # cv2.imshow('Python FPGA Annotation', frame)
                # cv2.waitKey(1000)

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--video-file", dest="video_file", default="", help="video file to be annotated")
    parser.add_option("", "--plot-body-part", dest="body_part", default="", help="video file to be annotated")
    parser.add_option("", "--meta-data-folder", dest="metaDataFolderString", default="", help="video file to be annotated")

    (options, args) = parser.parse_args()

    stampAnnotations(options.video_file, options.metaDataFolderString, options.body_part)