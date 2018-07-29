#!/usr/bin/env python

from optparse import OptionParser
import json
import re
import os
import numpy as np
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
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results.write visually, level 5 for all, level 1 for final, level 0 for none")

    (options, args) = parser.parse_args()

    time_lag = 5
    useCo = 1
    frameFiles_all = []
    ind_all = []
    error_loc_all = []

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            train_annotation = json.load(fin_annotation)
    else:
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                train_annotation = {}
                train_annotation_co = {}
                train_annotation['x'] = []
                train_annotation['y'] = []
                train_annotation['framefile'] = []
                train_annotation_co['x'] = []
                train_annotation_co['y'] = []
                train_annotation_co['framefile'] = []
                data_posterior_all = None
                X_all = None
                Y_all = None
                frameFiles_all = []

                with open(train_annotation_file) as fin_annotation:
                    tmp_train_annotation = json.load(fin_annotation)
                    for i in range(0, len(tmp_train_annotation["Annotations"])):
                        for j in range(0, len(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"])):
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == options.body_part):
                                train_annotation['x'].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation['y'].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                                train_annotation['framefile'].append(tmp_train_annotation["Annotations"][i]["FrameFile"])
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == 'RightMHhook'):
                                train_annotation_co['x'].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]))
                                train_annotation_co['y'].append(float(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]))
                                train_annotation_co['framefile'].append(tmp_train_annotation["Annotations"][i]["FrameFile"])

                train_annotation['x'] = np.array(train_annotation['x'])
                train_annotation['y'] = np.array(train_annotation['y'])

                train_annotation_co['x'] = np.array(train_annotation_co['x'])
                train_annotation_co['y'] = np.array(train_annotation_co['y'])

                data_posterior = []
                frameFiles = []
                data_posterior.append([])
                data_posterior.append([])
                data_posterior[0].append(train_annotation['x'][:-time_lag])
                data_posterior[1].append(train_annotation['y'][:-time_lag])
                frameFiles_all.extend(train_annotation['framefile'])
                data_posterior = np.squeeze(data_posterior)
                #print "data_posterior", np.shape(data_posterior)
                #kop = raw_input('Enter...')
                coData = []
                frameFiles = []
                coData.append([])
                coData.append([])
                coData[0].append(train_annotation_co['x'][:-2])
                coData[1].append(train_annotation_co['y'][:-2])
                coData = np.squeeze(coData)
                #print "coData", np.shape(coData)
                #kop = raw_input('Enter...')

                if useCo > 0:
                    data_posterior = np.hstack((data_posterior, coData))

                #print "data_posterior", np.shape(data_posterior)
                #kop = raw_input('Enter...')

                data_posterior = data_posterior.T

                k = 0
                for i in range(time_lag-1, 0, -1):
                    k += 1
                    temp = []
                    temp.append([])
                    temp.append([])
                    temp[0].append(train_annotation['x'][k:-i])
                    temp[1].append(train_annotation['y'][k:-i])
                    temp = np.squeeze(temp)
                    #print "Temp", np.shape(temp)
                    if useCo > 0:
                        temp_co = [train_annotation_co['x'][-i-2:-i],  train_annotation_co['y'][-i-2:-i]]
                        #print "Temp_co", np.shape(temp_co)
                        temp = np.hstack((temp, temp_co))

                    #print "Temp", np.shape(temp)
                    #raw_input("Press enter to continue")
                    temp = temp.T
                    data_posterior = np.hstack([data_posterior, temp])
                    # print "data_posterior", np.shape(data_posterior)
                    # raw_input("Press enter to continue")

                data_posterior = np.array(data_posterior)
                #print np.shape(data_posterior)
                X = train_annotation['x'][time_lag:]
                Y = train_annotation['y'][time_lag:]

                if useCo > 0:
                    X = np.hstack((X, train_annotation_co['x'][2:]))
                    Y = np.hstack((Y, train_annotation_co['y'][2:]))

                #print "X", np.shape(X)

                X_all = X
                Y_all = Y
                data_posterior_all = data_posterior
                # if ( data_posterior_all == None ):
                #     data_posterior_all = data_posterior
                # else:
                #     data_posterior_all = np.concatenate( (data_posterior_all, data_posterior), axis=0)
                # if ( X_all == None ):
                #     X_all = X
                # else:
                #     X_all = np.concatenate( (X_all, X), axis=0)
                # if ( Y_all == None ):
                #     Y_all = Y
                # else:
                #     Y_all = np.concatenate( (Y_all, Y), axis=0)

                error_thresh = 5

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

                error_loc = np.sqrt(X_diff*X_diff + Y_diff*Y_diff)

                ind_1 = np.where(error_loc <= error_thresh)
                ind_2 = np.where(X_all == -1.0)
                ind = np.intersect1d(ind_1, ind_2)
                ind = ind.tolist()
                ind = list(set(ind_1[0].tolist())-set(ind))

                ind_all.extend(ind)
                error_loc_all.extend(error_loc)
                if options.display_level > 0:
                    print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh, len(ind), len(error_loc), float(float(len(ind))/float(len(error_loc)))*100)
                    print "min localization error:", np.min(error_loc)
                    print "median localization error:", np.median(error_loc)
                    print "mean localization error:", np.mean(error_loc)
                    print "max localization error:", np.max(error_loc)

                if options.display_level > 0:
                    print "RANSAC"
                    for n in range(0, len(X_all)):
                        frame_file = frameFiles_all[n+time_lag-1]
                        frame_file = re.sub(".*/data/", "data/", frame_file)
                        frame_file = os.path.join(options.project_dir , frame_file)
                        frame = cv2.imread(frame_file)
                        cv2.circle(frame, (int(X_all[n]), int(Y_all[n])), 6, (255, 0, 0), thickness=-1)
                        cv2.circle(frame, (int(X_pred[n]), int(Y_pred[n])), 6, (0, 0, 255), thickness=-1)
                        display_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                        cv2.imshow("mouth hook annotation", display_frame)
                        key_press = cv2.waitKey(-1)
                        if (key_press == 113 or key_press == 13):
                            break

    print "\nAll Localization Errors"
    print "percentage inliers (<=%d pixels): %d/%d = %f" %(error_thresh, len(ind_all), len(error_loc_all),float(float(len(ind_all))/float(len(error_loc_all)))*100)
    print "min localization error:", np.min(error_loc_all)
    print "median localization error:", np.median(error_loc_all)
    print "mean localization error:", np.mean(error_loc_all)
    print "max localization error:", np.max(error_loc_all)