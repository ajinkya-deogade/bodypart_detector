#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation", dest="test_annotation_file", default="",help="frame level testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--training-data-file", dest="trained_data",help="File with information about training data")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.1,help="threshold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15,help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=3.0,help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--save-dir-error", dest="save_dir_error", default="", help="directory to save errors, if at all")

    (options, args) = parser.parse_args()

    with open(options.test_annotation_file) as fin_annotation:
        test_annotation = json.load(fin_annotation)

    surf = cv2.SURF(400, nOctaves=4, nOctaveLayers=4)
    class SaveClass:
        def __init__(self):
            self.votes = None
            self.keypoints = None
            self.descriptors = None
            self.bodypart = None

    error_file = {"num_outliers": 0,"outlier_error_thresh": 0,"proportion_outliers": 0,"median_error_dist": 0,"mean_error_dist": 0,"frames_evaluated": 0}
    rmh_knn = cv2.KNearest()

    rmh_trained_data = SaveClass()
    rmh_trained_data = pickle.load(open(options.trained_data, 'rb'))

    vote_train = rmh_trained_data.votes
    rmh_desc_train_samples = rmh_trained_data.descriptors
    rmh_kp_train_responses = rmh_trained_data.keypoints
    test_bodypart = rmh_trained_data.bodypart


    rmh_knn.train(rmh_desc_train_samples, rmh_kp_train_responses)
    rmh_vote = np.zeros((2 * options.vote_patch_size + 1, 2 * options.vote_patch_size + 1, 1), np.float)

    for x in range(-options.vote_patch_size, options.vote_patch_size + 1):
        for y in range(-options.vote_patch_size, options.vote_patch_size + 1):
            rmh_vote[y + options.vote_patch_size, x + options.vote_patch_size] = 1.0 + np.exp(
                -0.5 * (x * x + y * y) / (np.square(options.vote_sigma))) / (options.vote_sigma * np.sqrt(2 * np.pi))

    rmh_error_dists = []
    rmh_n_outlier_error_dist = 0
    rmh_n_eval = 0

    for i in range(0, len(test_annotation["Annotations"])):
        frame_file = test_annotation["Annotations"][i]["FrameFile"]
        frame_file = re.sub(".*/data/", "data/", frame_file)
        frame_file = os.path.join(options.project_dir, frame_file)
        #print frame_file

        frame = cv2.imread(frame_file)

        if (options.display_level >= 2):
            rmh_display_voters = frame.copy()

        rmh_coords_gt = None

        for j in range(0, len(test_annotation["Annotations"][i]["FrameValueCoordinates"])):
            if (test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == test_bodypart and test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                rmh_coords_gt = {}
                rmh_coords_gt["x"] = int(test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                rmh_coords_gt["y"] = int(test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

        rmh_vote_map = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 1), np.float)

        kp_frame, desc_frame = surf.detectAndCompute(frame, None)
        for h, desc in enumerate(desc_frame):
            desc = np.array(desc, np.float32).reshape((1, 128))
            retval, results, neigh_resp, dists = rmh_knn.find_nearest(desc, 1)
            r, d = int(results[0][0]), dists[0][0]
            if (d <= options.desc_distance_threshold):
                a = np.pi * kp_frame[h].angle / 180.0
                R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                p = kp_frame[h].pt + np.dot(R, vote_train[r])
                x, y = p
                if (not (x <= options.vote_patch_size or x >= np.shape(frame)[
                    1] - options.vote_patch_size or y <= options.vote_patch_size or y >= np.shape(frame)[
                    0] - options.vote_patch_size)):
                    rmh_vote_map[y - options.vote_patch_size:y + options.vote_patch_size + 1,
                    x - options.vote_patch_size:x + options.vote_patch_size + 1] += rmh_vote
                    if (options.display_level >= 2):
                        cv2.circle(display_voters, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)

        if (options.display_level >= 2):
            display_voters = cv2.resize(display_voters, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("voters", display_voters)

        vote_max = np.amax(rmh_vote_map)
        rmh_vote_map /= vote_max
        vote_max_loc = np.where(rmh_vote_map == np.amax(rmh_vote_map))

        rmh_coords_est = {}
        rmh_coords_est["x"] = int(vote_max_loc[1])
        rmh_coords_est["y"] = int(vote_max_loc[0])

        if (rmh_coords_gt is not None):
            rmh_error_dist = np.sqrt(np.square(rmh_coords_gt["x"] - rmh_coords_est["x"]) + np.square(rmh_coords_gt["y"] - rmh_coords_est["y"]))
            print "Distance between annotated and estimated RightMHhook location:", rmh_error_dist
            rmh_n_eval += 1
            if (rmh_error_dist <= options.outlier_error_dist):
                rmh_error_dists.append(rmh_error_dist)
            else:
                rmh_n_outlier_error_dist += 1

        if (options.display_level >= 1):
            display_vote_map = np.array(frame.copy(), np.float)
            display_vote_map /= 255.0
            display_vote_map[:, :, 2] = rmh_vote_map[:, :, 0]
            cv2.circle(display_vote_map, (rmh_coords_est["x"], rmh_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
            display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("voters", display_vote_map)

        if (options.save_dir_images != ""):
            display_vote_map = np.array(frame.copy(), np.float)
            display_vote_map /= 255.0
            display_vote_map[:, :, 2] = rmh_vote_map[:, :, 0]
            cv2.circle(display_vote_map, (rmh_coords_est["x"], rmh_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
            save_name=os.path.join(options.save_dir_images, os.path.splitext(os.path.basename(frame_file))[0]) + ".jpeg"
            cv2.imwrite(save_name, display_vote_map*255, (cv2.cv.CV_IMWRITE_JPEG_QUALITY,50))

        if (options.display_level >= 1):
            key_press = cv2.waitKey(-1)
            if (key_press == 113 or key_press == 13):
                break

    if (options.save_dir_error != ""):
        error_file = {"num_outliers": rmh_n_outlier_error_dist,"outlier_error_thresh": options.outlier_error_dist,"proportion_outliers": float(rmh_n_outlier_error_dist) / float(rmh_n_eval),"median_error_dist": np.median(rmh_error_dists),"mean_error_dist": np.mean(rmh_error_dists),"frames_evaluated": rmh_n_eval}
        with open(options.save_dir_error,'w') as fout_error:
             json.dump(error_file, fout_error, indent=4)

    print os.sys.argv
    print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (options.outlier_error_dist, rmh_n_outlier_error_dist, rmh_n_eval, float(rmh_n_outlier_error_dist) / float(rmh_n_eval) )
    print "Median inlier error dist =", np.median(rmh_error_dists)
    print "Mean inlier error dist =", np.mean(rmh_error_dists)

    cv2.destroyWindow("frame")
