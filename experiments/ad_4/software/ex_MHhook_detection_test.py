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
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list_fpgaKNNVal", default="",help="list of testing annotation JSON file")
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

    #with open(options.test_annotation_file) as fin_annotation:
        #test_annotation = json.load(fin_annotation)

    surf = cv2.SURF(400, nOctaves=4, nOctaveLayers=4)
    class SaveClass:
        def __init__(self):
            self.votes = None
            self.keypoints = None
            self.descriptors = None
            self.bodypart = None

    error_file = {"num_outliers": 0,"outlier_error_thresh": 0,"proportion_outliers": 0,"median_error_dist": 0,"mean_error_dist": 0,"frames_evaluated": 0}
    bodypart_knn = cv2.KNearest()

    bodypart_trained_data = SaveClass()
    bodypart_trained_data = pickle.load(open(options.trained_data, 'rb'))

    vote_train = bodypart_trained_data.votes
    bodypart_desc_train_samples = bodypart_trained_data.descriptors
    bodypart_kp_train_responses = bodypart_trained_data.keypoints
    test_bodypart = bodypart_trained_data.bodypart


    bodypart_knn.train(bodypart_desc_train_samples, bodypart_kp_train_responses)

    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
            print "Working on file: ",test_annotation_file
            bodypart_error_dists = []
            bodypart_n_outlier_error_dist = 0
            bodypart_n_eval = 0
            bodypart_vote = np.zeros((2 * options.vote_patch_size + 1, 2 * options.vote_patch_size + 1, 1), np.float)

            for x in range(-options.vote_patch_size, options.vote_patch_size + 1):
                for y in range(-options.vote_patch_size, options.vote_patch_size + 1):
                    bodypart_vote[y + options.vote_patch_size, x + options.vote_patch_size] = 1.0 + np.exp(
                        -0.5 * (x * x + y * y) / (np.square(options.vote_sigma))) / (options.vote_sigma * np.sqrt(2 * np.pi))

            for i in range(0, len(test_annotation["Annotations"])):
                frame_file = test_annotation["Annotations"][i]["FrameFile"]
                frame_file = re.sub(".*/data/", "data/", frame_file)
                frame_file = os.path.join(options.project_dir, frame_file)
                #print frame_file

                frame = cv2.imread(frame_file)

                if (options.display_level >= 2):
                    bodypart_display_voters = frame.copy()

                bodypart_coords_gt = None

                for j in range(0, len(test_annotation["Annotations"][i]["FrameValueCoordinates"])):
                    if (test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == test_bodypart and test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"] != -1 and test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"] != -1):
                        bodypart_coords_gt = {}
                        bodypart_coords_gt["x"] = int(test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"])
                        bodypart_coords_gt["y"] = int(test_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"])

                bodypart_vote_map = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 1), np.float)

                kp_frame, desc_frame = surf.detectAndCompute(frame, None)
                for h, desc in enumerate(desc_frame):
                    desc = np.array(desc, np.float32).reshape((1, 128))
                    retval, results, neigh_resp, dists = bodypart_knn.find_nearest(desc, 1)
                    r, d = int(results[0][0]), dists[0][0]
                    if (d <= options.desc_distance_threshold):
                        a = np.pi * kp_frame[h].angle / 180.0
                        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                        p = kp_frame[h].pt + np.dot(R, vote_train[r])
                        x, y = p
                        if (not (x <= options.vote_patch_size or x >= np.shape(frame)[1] - options.vote_patch_size or y <= options.vote_patch_size or y >= np.shape(frame)[0] - options.vote_patch_size)):
                            bodypart_vote_map[y - options.vote_patch_size:y + options.vote_patch_size + 1,
                            x - options.vote_patch_size:x + options.vote_patch_size + 1] += bodypart_vote
                            if (options.display_level >= 2):
                                cv2.circle(display_voters, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)

                if (options.display_level >= 2):
                    display_voters = cv2.resize(display_voters, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("voters", display_voters)

                vote_max = np.amax(bodypart_vote_map)
                bodypart_vote_map /= vote_max
                vote_max_loc = np.where(bodypart_vote_map == np.amax(bodypart_vote_map))

                bodypart_coords_est = {}
                bodypart_coords_est["x"] = int(vote_max_loc[1])
                bodypart_coords_est["y"] = int(vote_max_loc[0])

                if (bodypart_coords_gt is not None):
                    bodypart_error_dist = np.sqrt(np.square(bodypart_coords_gt["x"] - bodypart_coords_est["x"]) + np.square(bodypart_coords_gt["y"] - bodypart_coords_est["y"]))
                    print "Distance between annotated and estimated RightMHhook location:", bodypart_error_dist
                    bodypart_n_eval += 1
                    if (bodypart_error_dist <= options.outlier_error_dist):
                        bodypart_error_dists.append(bodypart_error_dist)
                    else:
                        bodypart_n_outlier_error_dist += 1

                if (options.display_level >= 1):
                    display_vote_map = np.array(frame.copy(), np.float)
                    display_vote_map /= 255.0
                    display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
                    cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
                    display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("voters", display_vote_map)

                if (options.save_dir_images != ""):
                    display_vote_map = np.array(frame.copy(), np.float)
                    display_vote_map /= 255.0
                    display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
                    cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
                    save_folder=os.path.join(options.save_dir_images, os.path.splitext(os.path.basename(test_annotation_file))[0])
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_name=os.path.join(save_folder,os.path.splitext(os.path.basename(frame_file))[0]) + ".jpeg"
                    cv2.imwrite(save_name, display_vote_map*255, (cv2.cv.CV_IMWRITE_JPEG_QUALITY,50))

                if (options.display_level >= 1):
                    key_press = cv2.waitKey(-1)
                    if (key_press == 113 or key_press == 13):
                        break

            if (options.save_dir_error != ""):
                error_content={"num_outliers": bodypart_n_outlier_error_dist,"outlier_error_thresh": options.outlier_error_dist,"proportion_outliers": float(bodypart_n_outlier_error_dist) / float(bodypart_n_eval),"median_error_dist": np.median(bodypart_error_dists),"mean_error_dist": np.mean(bodypart_error_dists),"frames_evaluated": bodypart_n_eval}
                if not os.path.exists(options.save_dir_error):
                        os.makedirs(options.save_dir_error)
                error_file= os.path.join(options.save_dir_error, os.path.splitext(os.path.basename(test_annotation_file))[0]) +'.json'
                with open(error_file,'w+') as fout_error:
                    json.dump(error_content, fout_error, indent=4)

    print os.sys.argv
    print "Number of outlier error distances (beyond %d) = %d / %d = %g" % (options.outlier_error_dist, bodypart_n_outlier_error_dist, bodypart_n_eval, float(bodypart_n_outlier_error_dist) / float(bodypart_n_eval) )
    print "Median inlier error dist =", np.median(bodypart_error_dists)
    print "Mean inlier error dist =", np.mean(bodypart_error_dists)

    cv2.destroyWindow("frame")
