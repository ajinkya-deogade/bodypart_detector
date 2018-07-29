#! /usr/bin/env python

import cv2
import numpy as np

if __name__ == '__main__':
    test_annotation_file = 'F:\MHDO_Tracking\data\Janelia_Q1_2014\RingLED\MPEG4\Extracted_Clips\\1_20140213R_s60_t1_clip_001.mp4'
    cap = cv2.VideoCapture(test_annotation_file)

    resol_width = int(cap.get(3))
    resol_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = int(cap.get(6))
    number_of_frames = int(cap.get(8))

    kernel = np.ones((15,15),np.uint8)

    while cap.isOpened():
        ret_read, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame,(5,5),0)
        ret,thresh = cv2.threshold(frame, 185, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(frame, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # thresh = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
        # closing = cv2.resize(closing, (0, 0), fx=0.5, fy=0.5)
        thresh = cv2.erode(thresh, kernel, iterations = 2)
        # thresh = cv2.resize(thresh, (0, 0), fx=0.5, fy=0.5)
        # cv2.imshow('Frame_Thresholded',closing)
        # cv2.imshow('Frame_Thresholded',thresh)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        abc = np.ones(np.shape(thresh))
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maxContourIndex = 0
        maxContourArea = 0
        i = 0
        while i<len(contours):
            areaC = cv2.contourArea(contours[i])
            if areaC > maxContourArea:
                maxContourIndex = i
            i +=1
        # print np.shape(contours[maxContourIndex])
        cnt = contours[maxContourIndex]
        cv2.drawContours(abc, [cnt], 0, (0, 255,0), 3)
        # # # print len(contours[1])
        abc = cv2.resize(abc, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Frame_Thresholded', abc)
        cv2.waitKey(100)
        # key_press = cv2.waitKey(-1)
        # if (key_press == 113 or key_press == 13):
        #     exit()
    cap.release()
    cv2.destroyAllWindows()