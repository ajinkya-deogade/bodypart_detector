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

    time_lag = 5
    useCo = 0
    error_loc_all = []
    nIter = 100
    percentageInliersAll = []

    for iter in range(0, nIter):
        print iter,
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
                    train_annotation_co = []
                    train_annotation_co.append([])
                    train_annotation_co.append([])
                    data_posterior_all = None
                    X_all = None
                    Y_all = None
                    with open(train_annotation_file) as fin_annotation:
                        tmp_train_annotation = json.load(fin_annotation)
                        for i in range(0, len(tmp_train_annotation["Annotations"])):
                            for j in range(0, len(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"])):
                                # if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "LeftDorsalOrgan"):
                                if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightDorsalOrgan"):
                                    train_annotation[0].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                    train_annotation[1].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                                # if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "LeftMHhook"):
                                if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightMHhook"):
                                    train_annotation_co[0].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                    train_annotation_co[1].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                    train_annotation = np.array(train_annotation)
                    train_annotation = train_annotation.T
                    train_annotation_co = np.array(train_annotation_co)
                    train_annotation_co = train_annotation_co.T

                    k = 0
                    data_posterior = train_annotation[:-time_lag] ## Exclude the last five elements
                    for i in range(time_lag-1, 0, -1):
                        k += 1
                        data_posterior = np.hstack([data_posterior, train_annotation[k:-i]])

                    if useCo:
                        data_posterior = np.hstack([data_posterior, train_annotation_co[time_lag:]])

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

                    error_thresh = 7
                    error_loc = []
                    j2 = []
                    try:
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
                        error_loc_all.append(error_loc)
                        percentageInliers.append(float(float(len(j2))/float(max(1, len(error_loc))))*100)
                    except:
                        error_loc_all.append(error_loc)
                        percentageInliers.append(float(float(len(j2))/float(max(1, len(error_loc))))*100)
                        continue

                    if options.display > 0:
                        print "RANSAC"
                        # print "error_loc", map(lambda x: round(x, 3), error_loc)
                        print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh,len(j2),len(error_loc),float(float(len(j2))/float(len(error_loc)))*100)
                        print "min localization error:", np.min(error_loc)
                        print "median localization error:", np.median(error_loc)
                        print "mean localization error:", np.mean(error_loc)
                        print "max localization error:", np.max(error_loc)

                percentageInliersAll.append(percentageInliers)

    error_loc_all = np.array(error_loc_all).T
    percentageInliersAll = np.array(percentageInliersAll)

    #     print "All Localizations Errors"
    #     j2 = [i for i in error_loc_all if i <= error_thresh]
    #     print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh, len(j2), len(error_loc_all), float(float(len(j2))/float(len(error_loc_all)))*100)
    #     print "min localization error:", np.min(error_loc_all)
    #     print "median localization error:", np.median(error_loc_all)
    #     print "mean localization error:", np.mean(error_loc_all)
    #     print "max localization error:", np.max(error_loc_all)
    #     percentageInliersAll.append(float(float(len(j2))/float(len(error_loc_all)))*100)

    if nIter > 1:
        # plt.boxplot(error_loc_all, showfliers=False)
        # plt.boxplot(percentageInliersAll, showfliers=False, bootstrap=True)
        sns.swarmplot(data=percentageInliersAll,color="white", edgecolor="gray")
        sns.boxplot(data=percentageInliersAll, whis=np.inf)
        # sns.stripplot(data=percentageInliersAll, color="white", edgecolor="gray", jitter=0.5)
        pylab.ylim(0, 100)
        pylab.ylabel('Percentage Inliers (<7 pixels)', fontsize=30)
        if useCo > 0:
            pylab.savefig('../expts/figures/swarmPlot_InferenceWithCo_Iter_100.png')
            pylab.savefig('../expts/figures/swarmPlot_InferenceWithCo_Iter_100.eps')
        else:
            pylab.savefig('../expts/figures/swarmPlot_InferenceWithoutCo_Iter_100.png')
            pylab.savefig('../expts/figures/swarmPlot_InferenceWithoutCo_Iter_100.eps')
        pylab.show()