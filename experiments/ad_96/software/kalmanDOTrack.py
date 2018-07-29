import cv2
import numpy as np
import pandas as pd
import pylab

trackerMetadataFile = '/Users/Ajinkya/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/InverseGaussian/Rawdata_20170303_201605/Rawdata_20170303_201605/Metadata_20170307_140356.txt'
trackerMetaData = pd.read_csv(trackerMetadataFile, delimiter=',', skiprows=0)
trackerMetaData = np.asarray(trackerMetaData)
ldoLocation = trackerMetaData[:, 18:20]
ldoConfidence = trackerMetaData[:, 47]

meas=[]
pred=[]

mp = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.003
kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 5

print np.shape(ldoLocation)
for i in range(3, np.shape(ldoLocation)[0]):
    if ldoConfidence[i] <= 20:
        continue

    mp = np.array((ldoLocation[i, 0], ldoLocation[i, 1]), np.float32)
    kalman.correct(mp)
    tp = kalman.predict()

    meas.append((int(mp[0]), int(mp[1])))
    pred.append((int(tp[0]),int(tp[1])))

    tempMeas = np.asarray(meas)
    tempPred = np.asarray(pred)

    timeIndex = []
    if i > 100:
        timeIndex = np.arange(np.shape(tempMeas)[0] - 50, np.shape(tempMeas)[0])
    else:
        timeIndex = np.arange(3, np.shape(tempMeas)[0])
        # pylab.plot(np.arange(3, i+1), tempMeas[:, 0], 'k-', np.arange(3, i+1), tempPred[:, 0], 'r-')
    if timeIndex != []:
        pylab.plot(timeIndex, tempMeas[timeIndex, 0], 'k-', timeIndex, tempPred[timeIndex, 0], 'r-')
        pylab.xlim((timeIndex[0], timeIndex[-1]))
        pylab.ylim((min(tempMeas[timeIndex, 0]) - 50, max(tempMeas[timeIndex, 0]) + 50))
        pylab.pause(0.05)
        pylab.cla