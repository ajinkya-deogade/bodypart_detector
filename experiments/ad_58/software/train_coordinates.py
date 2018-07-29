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

def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--annotation-list", dest="train_annotation_list", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--save-dir", dest="save_dir",default="MouthHook", help="Input the bodypart to be trained")

    (options, args) = parser.parse_args()

    # headers = ['FrameNumber','MouthHook_x','MouthHook_y','LeftMHhook_x','LeftMHhook_y','RightMHhook_x','RightMHhook_y','LeftDorsalOrgan_x','LeftDorsalOrgan_y','RightDorsalOrgan_x','RightDorsalOrgan_y']
    headers = ['MouthHook_x','MouthHook_y']
    time_lag = 5
    data_posterior_all = None
    X_all = None
    Y_all = None

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            train_annotation = json.load(fin_annotation)
    else:
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                train_annotation = []
                train_annotation.append([])
                train_annotation.append([])
                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    for i in range(0, len(tmp_train_annotation["Annotations"])):
                        for j in range(0, len(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"])):
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook"):
                                train_annotation[0].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation[1].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                train_annotation = np.array(train_annotation)
                train_annotation = train_annotation.T

                k = 0
                data_posterior = train_annotation[:-time_lag]
                print np.shape(data_posterior)
                for i in range(time_lag-1,0,-1):
                    k += 1
                    print np.shape(data_posterior)
                    data_posterior = np.hstack([data_posterior, train_annotation[k:-i]])

                print "data_posterior: ", np.shape(data_posterior)
                X = train_annotation[time_lag:,0]
                Y = train_annotation[time_lag:,1]

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

    print "data_posterior_all: ", np.shape(data_posterior_all)
    print "X_all: ", np.shape(X_all)
    print "train_annotation: ", len(train_annotation)
    error_thresh = 7

    print "Linear regression"
    X_fit, X_residuals, X_rank, X_sigular_val = np.linalg.lstsq(data_posterior_all, X_all)
    Y_fit, Y_residuals, Y_rank, Y_sigular_val = np.linalg.lstsq(data_posterior_all, Y_all)
    X_pred = np.dot(data_posterior_all, X_fit)
    X_diff = X_pred - X_all
    Y_pred = np.dot(data_posterior_all, Y_fit)
    Y_diff = Y_pred - Y_all
    error_loc = np.sqrt(X_diff*X_diff + Y_diff*Y_diff)
    j2 = [i for i in error_loc if i <= error_thresh]
    plt.plot(X_all, Y_all,'b.')
    plt.plot(X_pred, Y_pred,'r.')
    plt.show()

    print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh,len(j2),len(error_loc),float(float(len(j2))/float(len(error_loc)))*100)
    print "min localization error:", np.min(error_loc)
    print "median localization error:", np.median(error_loc)
    print "mean localization error:", np.mean(error_loc)
    print "max localization error:", np.max(error_loc)

    print "RANSAC"
    X_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=5, max_trials=100)
    Y_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=5, max_trials=100)
    X_model_ransac.fit(data_posterior_all, X_all)
    Y_model_ransac.fit(data_posterior_all, Y_all)
    X_pred = np.squeeze(X_model_ransac.predict(data_posterior_all))
    Y_pred = np.squeeze(Y_model_ransac.predict(data_posterior_all))
    X_diff = X_pred - X_all
    Y_diff = Y_pred - Y_all
    error_loc = np.sqrt(X_diff*X_diff + Y_diff*Y_diff)
    j2 = [i for i in error_loc if i <= error_thresh]

    print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh,len(j2),len(error_loc),float(float(len(j2))/float(len(error_loc)))*100)
    print "min localization error:", np.min(error_loc)
    print "median localization error:", np.median(error_loc)
    print "mean localization error:", np.mean(error_loc)
    print "max localization error:", np.max(error_loc)
