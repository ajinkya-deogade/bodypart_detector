#!/usr/bin/env python

from optparse import OptionParser
import json
import re
import os
import numpy as np
# import matplotlib.pyplot as plt
from  matplotlib import pylab
from sklearn import linear_model
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def string_split(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

if __name__ == '__main__':
    parser = OptionParser()

    # Read the options
    parser.add_option("", "--train-annotation", dest="train_annotation_file", default="", help="frame level training annotation JSON file")
    parser.add_option("", "--annotation-list", dest="train_annotation_list", default="",help="list of frame level training annotation JSON files")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--save-dir", dest="save_dir",default="MouthHook", help="Input the bodypart to be trained")
    parser.add_option("", "--display-level", dest="display",default=0, help="Visualize or not")

    (options, args) = parser.parse_args()

    time_lag = 6
    useCo = 0
    percentageInliers = []

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            train_annotation = json.load(fin_annotation)
    else:
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                frame_file = re.sub(".*/data/", "data/", train_annotation_file.strip())
                train_annotation_file = os.path.join(options.project_dir, frame_file)
                # train_annotation_file = os.path.join(options.project_dir, re.sub(".*/data/", "data/", train_annotation_file.strip()))
                train_annotation = []
                train_annotation.append([])
                train_annotation.append([])
                train_annotation_mh = []
                train_annotation_mh.append([])
                train_annotation_mh.append([])
                train_annotation_lmh = []
                train_annotation_lmh.append([])
                train_annotation_lmh.append([])
                train_annotation_rmh = []
                train_annotation_rmh.append([])
                train_annotation_rmh.append([])
                data_posterior = None
                X = None
                Y = None
                error_thresh = 6
                error_loc = []
                j2 = []

                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    for i in range(0, len(tmp_train_annotation["Annotations"])):
                        for j in range(0, len(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"])):
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightDorsalOrgan"):
                                train_annotation[0].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation[1].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightMHhook"):
                                train_annotation_rmh[0].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation_rmh[1].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "LeftMHhook"):
                                train_annotation_lmh[0].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation_lmh[1].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook"):
                                train_annotation_mh[0].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation_mh[1].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                train_annotation = np.array(train_annotation)
                train_annotation = train_annotation.T
                train_annotation_mh = np.array(train_annotation_mh)
                train_annotation_mh = train_annotation_mh.T
                train_annotation_lmh = np.array(train_annotation_lmh)
                train_annotation_lmh = train_annotation_lmh.T
                train_annotation_rmh = np.array(train_annotation_rmh)
                train_annotation_rmh = train_annotation_rmh.T
                X_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=2, max_trials=100)
                Y_model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold=7.0, min_samples=2, max_trials=100)
                predictedData = np.array(train_annotation[:time_lag+1])

                ## First Model
                data_posterior = np.hstack([train_annotation[:time_lag], train_annotation_mh[:time_lag], train_annotation_lmh[:time_lag], train_annotation_rmh[:time_lag]])
                X = data_posterior[:, 0]
                Y = data_posterior[:, 1]
                X_model_ransac.fit(data_posterior, X)
                Y_model_ransac.fit(data_posterior, Y)

                ## Loop over time
                for predictIndex in range(time_lag+1, len(train_annotation[:, 0])):
                    if predictIndex <= 2*time_lag:
                        # data_present_frame = train_annotation[predictIndex]
                        data_present_frame = np.hstack([train_annotation[predictIndex], train_annotation_mh[predictIndex], train_annotation_lmh[predictIndex], train_annotation_rmh[predictIndex]])
                        X_pred = np.squeeze(X_model_ransac.predict(data_present_frame))
                        Y_pred = np.squeeze(Y_model_ransac.predict(data_present_frame))
                        predictedData = np.vstack((predictedData, [X_pred, Y_pred]))

                        ## Create model for the next frame
                        data_posterior = np.hstack([train_annotation[predictIndex - time_lag:predictIndex], train_annotation_mh[predictIndex - time_lag:predictIndex], train_annotation_lmh[predictIndex - time_lag:predictIndex], train_annotation_rmh[predictIndex - time_lag:predictIndex]])
                        # data_posterior = train_annotation[predictIndex - time_lag:predictIndex]
                        X = data_posterior[:, 0]
                        Y = data_posterior[:, 1]
                        X_model_ransac.fit(data_posterior, np.reshape(X, (len(X), 1)))
                        Y_model_ransac.fit(data_posterior, np.reshape(Y, (len(Y), 1)))
                    else:
                        data_present_frame = np.hstack([train_annotation[predictIndex], train_annotation_mh[predictIndex], train_annotation_lmh[predictIndex], train_annotation_rmh[predictIndex]])
                        X_pred = np.squeeze(X_model_ransac.predict(data_present_frame))
                        Y_pred = np.squeeze(Y_model_ransac.predict(data_present_frame))
                        predictedData = np.vstack((predictedData, [X_pred, Y_pred]))

                        ## Create model for the next frame
                        data_posterior = np.hstack([predictedData[predictIndex - time_lag:predictIndex], train_annotation_mh[predictIndex - time_lag:predictIndex], train_annotation_lmh[predictIndex - time_lag:predictIndex], train_annotation_rmh[predictIndex - time_lag:predictIndex]])
                        # data_posterior = predictedData[predictIndex - time_lag:predictIndex] ## predicted data from now on
                        X = data_posterior[:, 0]
                        Y = data_posterior[:, 1]
                        X_model_ransac.fit(data_posterior, np.reshape(X, (len(X), 1)))
                        Y_model_ransac.fit(data_posterior, np.reshape(Y, (len(Y), 1)))
                    data_posterior = []

                indWithAnnotation = np.nonzero(train_annotation[:,0] != -1.0)
                X_diff = predictedData[indWithAnnotation, 0] - train_annotation[indWithAnnotation,0]
                Y_diff = predictedData[indWithAnnotation, 1] - train_annotation[indWithAnnotation,1]
                error_loc = np.squeeze(np.sqrt(X_diff*X_diff + Y_diff*Y_diff))
                inliers = error_loc[np.nonzero(error_loc <= error_thresh)]

                if options.display > -1:
                    print "RANSAC"
                    if len(error_loc) > 0:
                        print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh,len(inliers),len(error_loc),float(float(len(inliers))/float(len(error_loc)))*100)
                        print "min localization error:", np.min(error_loc)
                        print "median localization error:", np.median(error_loc)
                        print "mean localization error:", np.mean(error_loc)
                        print "max localization error:", np.max(error_loc)
                        raw_input("Press Enter.....")