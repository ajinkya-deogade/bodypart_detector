#!/usr/bin/env python

from optparse import OptionParser
import json
import re
import csv
import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import linear_model
import cv2

def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--annotation-list", dest="train_annotation_list", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--save-dir", dest="save_dir",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--body-part", dest="body_part",default="MouthHook", help="Input the bodypart to be trained")

    (options, args) = parser.parse_args()

    time_lag = 5

    data_posterior_all = None
    X_all = None
    Y_all = None
    frameFiles_all = []

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            train_annotation = json.load(fin_annotation)
    else:
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                train_annotation = {}
                train_annotation['x'] = []
                train_annotation['y'] = []
                train_annotation['framefile'] = []

                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)

                    for i in range(0, len(tmp_train_annotation["Annotations"])):
                        for j in range(0, len(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"])):
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == options.body_part):
                                train_annotation['x'].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation['y'].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                                train_annotation['framefile'].append(tmp_train_annotation["Annotations"][i]["FrameFile"])

                train_annotation['x'] = np.array(train_annotation['x'])
                train_annotation['y'] = np.array(train_annotation['y'])
                print 'Len: ', np.shape(train_annotation['framefile'])

                k = 0
                data_posterior = []
                frameFiles = []
                data_posterior.append([])
                data_posterior.append([])
                data_posterior[0].append(train_annotation['x'][:-time_lag])
                data_posterior[1].append(train_annotation['y'][:-time_lag])
                frameFiles_all.extend(train_annotation['framefile'])
                data_posterior = np.squeeze(data_posterior)
                data_posterior = data_posterior.T
                print "All Frame Files: ", len(frameFiles_all)

                for i in range(time_lag-1, 0, -1):
                    k += 1
                    temp = []
                    temp.append([])
                    temp.append([])
                    temp[0].append(train_annotation['x'][k:-i])
                    temp[1].append(train_annotation['y'][k:-i])
                    temp = np.squeeze(temp)
                    temp = temp.T
                    data_posterior = np.hstack([data_posterior, temp])

                data_posterior = np.array(data_posterior)
                # data_posterior = data_posterior.T

                X = train_annotation['x'][time_lag:]
                Y = train_annotation['y'][time_lag:]

                if ( data_posterior_all == None ):
                    data_posterior_all = data_posterior
                else:
                    data_posterior_all = np.concatenate( (data_posterior_all, data_posterior), axis=0)
                if ( X_all == None ):
                    X_all = X
                else:
                    X_all = np.concatenate( (X_all, X), axis=0)
                if ( Y_all == None ):
                    Y_all = Y
                else:
                    Y_all = np.concatenate( (Y_all, Y), axis=0)

    error_thresh = 5

    print "Len train_annotation: ", len(train_annotation['x'])
    print "Len frameFiles: ", len(frameFiles_all)
    # print "Len train_annotation: ", np.shape(data_posterior_all)

    print "RANSAC"
    X_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=5, max_trials=100)
    Y_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=5, max_trials=100)
    X_model_ransac.fit(data_posterior_all, X_all)
    Y_model_ransac.fit(data_posterior_all, Y_all)
    X_pred = np.squeeze(X_model_ransac.predict(data_posterior_all))
    Y_pred = np.squeeze(Y_model_ransac.predict(data_posterior_all))
    X_pred = map(round, X_pred)
    Y_pred = map(round, Y_pred)
    X_diff = X_pred - X_all
    Y_diff = Y_pred - Y_all
    print "Len Pred: ", len(X_pred)
    print X_pred, "\n"
    print X_all, "\n"
    error_loc = np.sqrt(X_diff*X_diff + Y_diff*Y_diff)
    j2 = [i for i in error_loc if i <= error_thresh]

    ind_1 = np.where(error_loc <= error_thresh)
    ind_2 = np.where(X_all == -1.0)
    ind = np.intersect1d(ind_1, ind_2)
    ind = ind.tolist()
    ind = list(set(ind_1[0].tolist())-set(ind))
    print len(ind)

    # for n in ind:
    for n in range(0, len(X_all)):
        frame_file = frameFiles_all[n+time_lag-1]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir , frame_file)
        # print frame_file

        frame = cv2.imread(frame_file)
        cv2.circle(frame, (int(X_all[n]), int(Y_all[n])), 6, (255, 0, 0), thickness=-1)
        cv2.circle(frame, (int(X_pred[n]), int(Y_pred[n])), 6, (0, 0, 255), thickness=-1)
        display_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("mouth hook annotation", display_frame)
        key_press = cv2.waitKey(-1)
        if (key_press == 113 or key_press == 13):
            break

    print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh, len(ind), len(error_loc),float(float(len(ind))/float(len(error_loc)))*100)
    print "min localization error:", np.min(error_loc)
    print "median localization error:", np.median(error_loc)
    print "mean localization error:", np.mean(error_loc)
    print "max localization error:", np.max(error_loc)