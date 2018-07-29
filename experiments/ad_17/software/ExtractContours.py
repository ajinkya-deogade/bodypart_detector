#! /usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

def perimeterCurvature(cnt, fitRes, curvature, curveDist):
    for i in range(0, fitRes):
        if (i < curveDist):
            curvature[i] = np.arctan2(cnt[i + curveDist, 0, 1] - cnt[i, 0, 1],
                                      cnt[i + curveDist, 0, 0] - cnt[i, 0, 0]) - np.arctan2(
                cnt[fitRes - (curveDist - i), 0, 1] - cnt[i, 0, 1],
                cnt[fitRes - (curveDist - i), 0, 0] - cnt[i, 0, 0])
        elif (i > (fitRes - curveDist - 1)):
            curvature[i] = np.arctan2(cnt[curveDist - (fitRes - i), 0, 1] - cnt[i, 0, 1],
                              cnt[curveDist - (fitRes - i), 0, 0] - cnt[i, 0, 0]) - np.arctan2(
            cnt[i - curveDist, 0, 1] - cnt[i,0,1], cnt[i - curveDist, 0, 0] - cnt[i, 0, 0])
        else:
            curvature[i] = np.arctan2(cnt[i + curveDist, 0, 1] - cnt[i, 0, 1],
                          cnt[i + curveDist, 0, 0] - cnt[i, 0, 0]) - np.arctan2(
        cnt[i - curveDist, 0, 1] - cnt[i, 0, 1], cnt[i - curveDist, 0, 0] - cnt[i, 0, 0])

        # if (curvature[i] < 0):
        #     curvature[i] = curvature[i] + 2 * np.pi;

    return True

if __name__ == '__main__':
    test_annotation_file = 'F:\MHDO_Tracking\data\Janelia_Q1_2014\RingLED\MPEG4\Extracted_Clips\\1_20140213R_s60_t1_clip_001.mp4'
    cap = cv2.VideoCapture(test_annotation_file)

    resol_width = int(cap.get(3))
    resol_height = int(cap.get(4))
    fps = int(cap.get(5))
    fourcc = int(cap.get(6))
    number_of_frames = int(cap.get(8))

    kernel = np.ones((15, 15), np.uint8)

    while cap.isOpened():
        ret_read, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        ret, thresh = cv2.threshold(frame, 185, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(frame, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # thresh = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
        # closing = cv2.resize(closing, (0, 0), fx=0.5, fy=0.5)
        thresh = cv2.erode(thresh, kernel, iterations=2)
        # thresh = cv2.resize(thresh, (0, 0), fx=0.5, fy=0.5)
        # cv2.imshow('Frame_Thresholded',closing)
        # cv2.imshow('Frame_Thresholded',thresh)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # abc = np.ones(np.shape(thresh))
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maxContourIndex = 0
        maxContourArea = 0
        i = 0
        while i < len(contours):
            areaC = cv2.contourArea(contours[i])
            if areaC > maxContourArea:
                maxContourIndex = i
            i += 1
        # print np.shape(contours[maxContourIndex])
        cnt = contours[maxContourIndex]
        fitRes = len(cnt)
        curvature = np.ones(shape=(fitRes,1))
        curveDist = 1
        result = perimeterCurvature(cnt, fitRes, curvature, curveDist)
        print result
        # print curvature
        plt.plot(curvature)
        # contourX = cnt[:,0,0]
        # contourY = cnt[:,0,1]
        # print kh[0,1]
        # cv2.drawContours(abc, [crv], 0, (0, 255,0), 3)
        # print len(cnt)
        # abc = cv2.resize(abc, (0, 0), fx=0.5, fy=0.5)
        # cv2.imshow('Frame_Thresholded', abc)
        # cv2.waitKey(100)
        # key_press = cv2.waitKey(-1)
        # if (key_press == 113 or key_press == 13):
        # exit()
    cap.release()
    cv2.destroyAllWindows()



    #
    # void AnalysisModule::findHeadTail(std::vector<cv::Point2f> & cFit, std::vector<double> crv, std::vector<cv::Point2f> & headTail){
    #
    # 	//Smallest (sharpest internal angle) curvature point is head (typically), suppress values around this point.
    # 	//Once one end is clearly much larger in curvature, track that one as the head using proximity measures.
    # 	//Next minimum curvature point is tail, bends in body have positive curvature, so don't get picked up
    #
    # 	//find first minimum of curvature (sharpest point is typically the head)
    # 	int minInd=0; double minVal=crv[0];
    # 	for i in range(0,crv.size()):
    # 		if(crv[i] < minVal){ minVal = crv[i]; minInd = i;}
    #
    # 	headTail.resize(2);
    # 	headTail[0] = cFit[minInd];
    #
    # 	//ignore values around this min and find the next minimum (typically tail)
    # 	//this loop does not modify the variable curve handed in since it passes by copy
    # 	int suppress = fitRes/8;  //suppress 25% of the perimeter
    # 	for (int i= minInd-suppress; i<(minInd+suppress); i++){
    # 		if (i<0)				crv[fitRes+i] = 600;
    # 		if (i>=fitRes)			crv[i-fitRes] = 600;
    # 		if((i>=0) & (i<fitRes))	crv[i] = 600;
    # 	}
    #
    # 	//find next min
    # 	minInd=0; minVal=crv[0];
    # 	for(int i=0; i<crv.size(); i++)
    # 		if(crv[i] < minVal){ minVal = crv[i]; minInd = i;}
    #
    # 	headTail[1] = cFit[minInd];
    #
    # 	//Determine which is head and which is tail based on long term curvature min and forward walking
    # 	if(index == -1){
    # 		masterHeadTail = headTail;
    # 		votesHT[0]++;
    # 	}else{
    # 		//Determine proximities to detect if a flip occurred
    # 		distHT[0] = pow(masterHeadTail[0].x - headTail[0].x,2) + pow(masterHeadTail[0].y - headTail[0].y,2);
    # 		distHT[1] = pow(masterHeadTail[0].x - headTail[1].x,2) + pow(masterHeadTail[0].y - headTail[1].y,2);
    # 		distHT[2] = pow(masterHeadTail[1].x - headTail[0].x,2) + pow(masterHeadTail[1].y - headTail[0].y,2);
    # 		distHT[3] = pow(masterHeadTail[1].x - headTail[1].x,2) + pow(masterHeadTail[1].y - headTail[1].y,2);
    #
    # 		minInd = 0; //find minimum
    # 		for(int i=0; i<4; i++) if(distHT[i] < distHT[minInd]) minInd = i;
    #
    # 		//Update the coordinates of the proximity tracked points
    # 		if(minInd == 0 | minInd == 3){//No flip has occurred
    # 			masterHeadTail = headTail;
    # 			votesHT[0]++;
    # 		}else{
    # 			//flip them into the initial head tail, give a vote to second entry that was not initially sharpest
    # 			masterHeadTail[0] = headTail[1];
    # 			masterHeadTail[1] = headTail[0];
    # 			votesHT[1]++;
    # 		}
    #
    # 		//Also add votes to the point in the direction of motion when the head/tail angle is small
    # 		//(assumes the animal goes mostly forward when walking)
    # 		if(index>-1){ //need history to calculate a velocity direction
    # 			//Currently not implemented
    # 		}
    # 	}
    #
    # 	//Assign based on votes
    # 	//The master point with the highest number of votes is placed first in the return list to represent the head
    # 	if(votesHT[0]>= votesHT[1]){
    # 		headTail = masterHeadTail;
    # 	}else{
    # 		headTail[0] = masterHeadTail[1];
    # 		headTail[1] = masterHeadTail[0];
    # 	}
    #
    #
    # };