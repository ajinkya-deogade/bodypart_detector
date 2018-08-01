# #! /opt/local/bin/python
#!/usr/bin/env python2

import re
from optparse import OptionParser
from pyflann import *
import cv2
import numpy as np
import pandas as pd
import glob, os

def stampAnnotations(video_file, project_dir, plot_body_part):

    video_file = re.sub(".*/data/", "data/", video_file)
    video_file = os.path.abspath(os.path.join(project_dir, video_file))

    head, tail = os.path.split(video_file)
    print os.path.join(os.path.abspath(head), os.path.splitext(tail)[0])
    trackerMetadataFile = [name for name in os.listdir(os.path.join(os.path.abspath(head), os.path.splitext(tail)[0]))]
    trackerMetadataFile = os.path.join(os.path.abspath(head), os.path.splitext(tail)[0], trackerMetadataFile[0])
    print trackerMetadataFile

    print "Video File: ", video_file
    cap = cv2.VideoCapture(video_file)

    numberFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print 'Number of Video Frames Recorded : ', numberFrames

    print trackerMetadataFile
    trackerMetaData = pd.read_csv(trackerMetadataFile, delimiter=',', skiprows=0)
    trackerMetaData = np.asarray(trackerMetaData)
    numberFramesTrackerRecorded = np.shape(trackerMetaData)[0]
    print 'Number of Metadata Frames Recorded : ', numberFramesTrackerRecorded

    crop_size = 512
    confThreshold = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    outputFrameSize = np.multiply(np.ones((512, 512, 3), dtype=np.uint8), 255)
    height, width, layers = outputFrameSize.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    bodyPartString = ''
    for bp in plot_body_part:
        bodyPartString = bodyPartString + '_' + bp
    bodyPartString = bodyPartString[1:]
    outputVideoFile = os.path.join(head, os.path.splitext(tail)[0], os.path.splitext(tail)[0]) + '_%s.avi' %(bodyPartString)
    video = cv2.VideoWriter(outputVideoFile, 0, 5, (width, height))
    AllDetectedBodyPart = []

    for row in range(0, numberFramesTrackerRecorded):
        detectedBodyPart = {}
        if ~(np.any(np.isnan(trackerMetaData[row, :]))):
            detectedBodyPart['Head'] = {}
            detectedBodyPart['Head']['x'], detectedBodyPart['Head']['y'] = int(trackerMetaData[row, 8]), int(trackerMetaData[row, 9])

            detectedBodyPart['Tail'] = {}
            detectedBodyPart['Tail']['x'], detectedBodyPart['Tail']['y'] = int(trackerMetaData[row, 10]), int(trackerMetaData[row, 11])

            detectedBodyPart['MidPoint'] = {}
            detectedBodyPart['MidPoint']['x'], detectedBodyPart['MidPoint']['y'] = int(trackerMetaData[row, 6]), int(trackerMetaData[row, 7])

            detectedBodyPart['MouthHook'] = {}
            detectedBodyPart['MouthHook']['x'], detectedBodyPart['MouthHook']['y'] = int(trackerMetaData[row, 12]), int(trackerMetaData[row, 13])
            detectedBodyPart['MouthHook']['conf'] = int(trackerMetaData[row, 44])
            detectedBodyPart['MouthHook']['spot_size'] = 20

            detectedBodyPart['LeftMHhook'] = {}
            detectedBodyPart['LeftMHhook']['x'], detectedBodyPart['LeftMHhook']['y'] = int(trackerMetaData[row, 14]), int(trackerMetaData[row, 15])
            detectedBodyPart['LeftMHhook']['conf'] = int(trackerMetaData[row, 45])
            detectedBodyPart['LeftMHhook']['spot_size'] = 20

            detectedBodyPart['RightMHhook'] = {}
            detectedBodyPart['RightMHhook']['x'], detectedBodyPart['RightMHhook']['y'] = int(trackerMetaData[row, 16]), int(trackerMetaData[row, 17])
            detectedBodyPart['RightMHhook']['conf'] = int(trackerMetaData[row, 46])
            detectedBodyPart['RightMHhook']['spot_size'] = 20

            detectedBodyPart['LeftDorsalOrgan'] = {}
            detectedBodyPart['LeftDorsalOrgan']['x'], detectedBodyPart['LeftDorsalOrgan']['y'] = int(trackerMetaData[row, 18]), int(trackerMetaData[row, 19])
            detectedBodyPart['LeftDorsalOrgan']['conf'] = int(trackerMetaData[row, 47])
            detectedBodyPart['LeftDorsalOrgan']['spot_size'] = 20

            detectedBodyPart['RightDorsalOrgan'] = {}
            detectedBodyPart['RightDorsalOrgan']['x'], detectedBodyPart['RightDorsalOrgan']['y'] = int(trackerMetaData[row, 20]), int(trackerMetaData[row, 21])
            detectedBodyPart['RightDorsalOrgan']['conf'] = int(trackerMetaData[row, 48])
            detectedBodyPart['RightDorsalOrgan']['spot_size'] = 20

            detectedBodyPart['FrameNumber'] = int(trackerMetaData[row, 0])
            AllDetectedBodyPart.append(detectedBodyPart)

    print 'Number of Correct Frames Recorded : ', np.shape(AllDetectedBodyPart)[0]

    frameNumber = 0
    for frameInfo in AllDetectedBodyPart:
        actualIndex = frameInfo['FrameNumber']
        # print 'Actual Index : ', actualIndex
        cropCenter_X = frameInfo['Head']['x']
        cropCenter_Y = frameInfo['Head']['y']

        if cap.isOpened():
            try:
                frameNumber += 1
                # cap.set(1, actualIndex - 1)
                cap.set(1, frameNumber - 1)
                # print 'Actual Frame : ', actualIndex
                ret, originalFrame = cap.read()
            except:
                print 'Not able read frame %d' % (frameNumber)
                continue

            frame = originalFrame.copy()
            frameOverlay = originalFrame.copy()
            crop_x = max(0, cropCenter_X-int(crop_size/2))
            crop_y = max(0, cropCenter_Y-int(crop_size/2))

            for bp in plot_body_part:
            # if fpgaBodyPartInfo['conf'] >= confThreshold:
                spotSize = frameInfo[bp]['spot_size']
                recTopLeft = {}
                recTopLeft['x'] = int(frameInfo[bp]['x']) - int(spotSize/2)
                recTopLeft['y'] = int(frameInfo[bp]['y']) - int(spotSize/2)

                recBottomRight = {}
                recBottomRight['x'] = int(frameInfo[bp]['x']) + int(spotSize/2)
                recBottomRight['y'] = int(frameInfo[bp]['y']) + int(spotSize/2)

                if bp == 'LeftDorsalOrgan':
                    cv2.rectangle(frameOverlay, (recTopLeft['x'], recTopLeft['y']), (recBottomRight['x'], recBottomRight['y']), (125, 125, 255), thickness=-1)
                    cv2.putText(frame, '%s: Conf %d Spot %dum' % ('LDO', frameInfo[bp]['conf'], spotSize*2.75), (crop_x + 320, crop_y + 60), font, 0.5, (125, 125, 255), 2)
                elif bp == 'RightDorsalOrgan':
                    cv2.rectangle(frameOverlay, (recTopLeft['x'], recTopLeft['y']), (recBottomRight['x'], recBottomRight['y']), (0, 255, 0), thickness=-1)
                    cv2.putText(frame, '%s: Conf %d Spot %dum' % ('RDO', frameInfo[bp]['conf'], spotSize*2.75), (crop_x + 320, crop_y + 80), font, 0.5, (0, 255, 0), 2)
                elif bp == 'MouthHook':
                    cv2.rectangle(frameOverlay, (recTopLeft['x'], recTopLeft['y']), (recBottomRight['x'], recBottomRight['y']), (255, 0, 0), thickness=-1)
                    cv2.putText(frame, '%s: Conf %d Spot %dum' % ('MH', frameInfo[bp]['conf'], spotSize*2.75), (crop_x + 320, crop_y + 40), font, 0.5, (255, 0, 0), 2)
                elif bp == 'Head':
                    cv2.rectangle(frameOverlay, (recTopLeft['x'], recTopLeft['y']), (recBottomRight['x'], recBottomRight['y']), (255, 0, 0), thickness=-1)
                    cv2.putText(frame, '%s: Spot %dum' % ('Head', spotSize * 2.75), (crop_x + 320, crop_y + 40), font, 0.5, (255, 0, 0), 2)

            frame = frame[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size, :]
            frameOverlay = frameOverlay[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :]
            alpha = 0.3
            cv2.addWeighted(frameOverlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, str(frameNumber), (40, 40), font, 0.5, (255, 255, 255), 2)

            # video.write(frame)
            cv2.imshow('Python FPGA Annotation', frame)
            cv2.waitKey(1000)

    cv2.destroyAllWindows()
    video.release()

def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--video-file-list", dest="video_file_list", default="", help="video file to be annotated")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--start-frame", dest="start_frame", default=0, type="int", help="path containing data directory")
    parser.add_option("", "--plot-body-part", dest="body_part", default="MouthHook", type="string", help="video file to be annotated", action="callback", callback=string_split)

    (options, args) = parser.parse_args()

    tempFile = []
    with open(options.video_file_list) as all_list:
        for vidFile in all_list:
            tempFile.append(vidFile)
    video_file = tempFile[0]

    stampAnnotations(video_file, options.project_dir, options.body_part)