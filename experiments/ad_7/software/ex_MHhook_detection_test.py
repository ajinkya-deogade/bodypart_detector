#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle
<<<<<<< HEAD
import time
=======
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--positive-training-datafile", dest="train_data_p", help="File to save the information about the positive training data")
    parser.add_option("", "--negative-training-datafile", dest="train_data_n", help="File to save the information about the negative training data")
    parser.add_option("", "--desc-dist-threshold", dest="desc_distance_threshold", type="float", default=0.1,help="threshold on distance between test descriptor and its training nearest neighbor to count its vote")
    parser.add_option("", "--vote-patch-size", dest="vote_patch_size", type="int", default=15,help="half dimension of the patch within which each test descriptor casts a vote, the actual patch size is 2s+1 x 2s+1")
    parser.add_option("", "--vote-sigma", dest="vote_sigma", type="float", default=3.0,help="spatial sigma spread of a vote within the voting patch")
    parser.add_option("", "--outlier-error-dist", dest="outlier_error_dist", type="int", default=15,help="distance beyond which errors are considered outliers when computing average stats")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    parser.add_option("", "--save-dir-images", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
<<<<<<< HEAD
    parser.add_option("", "--video-file", dest="video_file", default="", help="path of the video file")
    (options, args) = parser.parse_args()

    surf = cv2.SURF(400, nOctaves=4, nOctaveLayers=4)

=======

    (options, args) = parser.parse_args()

    surf = cv2.SURF(400, nOctaves=4, nOctaveLayers=4)
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7
    class SaveClass:
        def __init__(self):
            self.votes = None
            self.keypoints = None
            self.descriptors = None
            self.bodypart = None

    error_file = {"num_outliers": 0,"outlier_error_thresh": 0,"proportion_outliers": 0,"median_error_dist": 0,"mean_error_dist": 0,"frames_evaluated": 0}
    bodypart_knn_pos = cv2.KNearest()
    bodypart_knn_neg = cv2.KNearest()

    bodypart_trained_data_pos = SaveClass()
    bodypart_trained_data_pos = pickle.load(open(options.train_data_p, 'rb'))
    bodypart_trained_data_neg = SaveClass()
    bodypart_trained_data_neg = pickle.load(open(options.train_data_n, 'rb'))

    vote_train_pos = bodypart_trained_data_pos.votes
    bodypart_desc_train_samples_pos = bodypart_trained_data_pos.descriptors
    bodypart_kp_train_responses_pos = bodypart_trained_data_pos.keypoints

    vote_train_neg = bodypart_trained_data_neg.votes
    bodypart_desc_train_samples_neg = bodypart_trained_data_neg.descriptors
    bodypart_kp_train_responses_neg = bodypart_trained_data_neg.keypoints
    test_bodypart = bodypart_trained_data_neg.bodypart

    bodypart_knn_pos.train(bodypart_desc_train_samples_pos, bodypart_kp_train_responses_pos)
    bodypart_knn_neg.train(bodypart_desc_train_samples_neg, bodypart_kp_train_responses_neg)
    bodypart_vote = np.zeros((2 * options.vote_patch_size + 1, 2 * options.vote_patch_size + 1, 1), np.float)
<<<<<<< HEAD
    # video_fi = 'Desktop/Rawdata_20140722_154715.mp4'
    cap = cv2.VideoCapture(options.video_file)
    print cap
=======

    test_annotation_file = '/Volumes/HD2/MHDO_Tracking/data/Janelia_Q1_2014/RingLED/MPEG4/9_20140213R.mp4'
    cap = cv2.VideoCapture(test_annotation_file)

>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7
    # resol_width = int(cap.get(3))
    # resol_height = int(cap.get(4))
    # fps = int(cap.get(5))
    # fourcc = int(cap.get(6))
   # out = cv2.VideoWriter()
    # success = out.open('~/Desktop/1_20140214R_1_Prediction.mp4',fourcc,fps,(1920,1920),True)
    # print success
    n = 0
<<<<<<< HEAD
    sum_total = 0

    while (cap.isOpened() and n <= 2000):
=======

    while (cap.isOpened()):
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7
        ret, frame = cap.read()
        print ret
        if ret is False:
            print 'Skipped Frame', n
<<<<<<< HEAD
            n += 1
            continue
        else:
            loop_start_time = time.time()
            n += 1
            print n
=======
            n = n+1
            continue
        else:
            n = n+1
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7
            for x in range(-options.vote_patch_size, options.vote_patch_size + 1):
                for y in range(-options.vote_patch_size, options.vote_patch_size + 1):
                    bodypart_vote[y + options.vote_patch_size, x + options.vote_patch_size] = 1.0 + np.exp(-0.5 * (x * x + y * y) / (np.square(options.vote_sigma))) / (options.vote_sigma * np.sqrt(2 * np.pi))

            if (options.display_level >= 2):
                display_voters = frame.copy()

            bodypart_coords_gt = None
            bodypart_vote_map = np.zeros((np.shape(frame)[0], np.shape(frame)[1], 1), np.float)
            kp_frame, desc_frame = surf.detectAndCompute(frame, None)

            for h, desc in enumerate(desc_frame):
                desc = np.array(desc, np.float32).reshape((1, 128))
                retval_pos, results_pos, neigh_resp_pos, dists_pos = bodypart_knn_pos.find_nearest(desc, 1)
                retval_neg, results_neg, neigh_resp_neg, dists_neg = bodypart_knn_neg.find_nearest(desc, 1)
                r_pos, d_pos = int(results_pos[0][0]), dists_pos[0][0]
                r_neg, d_neg = int(results_neg[0][0]), dists_neg[0][0]
                relative_distance = d_pos - d_neg
                if (relative_distance <= options.desc_distance_threshold):
                    a = np.pi * kp_frame[h].angle / 180.0
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    p = kp_frame[h].pt + np.dot(R, vote_train_pos[r_pos])
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
<<<<<<< HEAD
            if ( vote_max > 0 and vote_max > frame_vote_max ):
                frame_vote_max = vote_max
                vote_max_loc = np.array(np.where(bodypart_vote_map == vote_max))
                vote_max_loc = vote_max_loc[:,0]
                bodypart_coords_est["x"] = int(vote_max_loc[1]) + int(image_header["crop_x"])
                bodypart_coords_est["y"] = int(vote_max_loc[0]) + int(image_header["crop_y"])
            else:
                bodypart_coords_est = None
=======
            bodypart_vote_map /= vote_max
            vote_max_loc = np.where(bodypart_vote_map == np.amax(bodypart_vote_map))

            bodypart_coords_est = {}
            bodypart_coords_est["x"] = int(vote_max_loc[1])
            bodypart_coords_est["y"] = int(vote_max_loc[0])

            if (options.display_level >= 1):
                display_vote_map = np.array(frame.copy(), np.float)
                display_vote_map /= 255.0
                display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
                cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
                # display_vote_map = cv2.resize(display_vote_map, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("voters", display_vote_map)
                h1, w1, d1 = display_vote_map.shape
                print "Frame Size: ", h1, w1, d1
                display_vote_map = display_vote_map * 255
                # out.write(display_vote_map)
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7

            if (options.save_dir_images != ""):
                display_vote_map = np.array(frame.copy(), np.float)
                display_vote_map /= 255.0
                display_vote_map[:, :, 2] = bodypart_vote_map[:, :, 0]
                cv2.circle(display_vote_map, (bodypart_coords_est["x"], bodypart_coords_est["y"]), 4, (0, 255, 255), thickness=-1)
<<<<<<< HEAD
                save_folder=os.path.join(options.save_dir_images, os.path.splitext(os.path.basename(options.video_file))[0])
=======
                save_folder=os.path.join(options.save_dir_images, os.path.splitext(os.path.basename(test_annotation_file))[0])
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                save_name=os.path.join(save_folder,str(n)) + ".jpeg"
                cv2.imwrite(save_name, display_vote_map*255, (cv2.cv.CV_IMWRITE_JPEG_QUALITY,50))
<<<<<<< HEAD
        loop_end_time = time.time()
        total_time = loop_end_time - loop_start_time
        sum_total += total_time
        print 'Loop %d : %d' % (n,total_time)
    print 'Total Time %d frames = %d' %(n, sum_total)
    print 'Average Time per frame = %d / %d = %g ' %(sum_total, n, float(sum_total)/float(n))
=======

            if (options.display_level >= 1):
                key_press = cv2.waitKey(-1)
                if (key_press == 113 or key_press == 13):
                    break
>>>>>>> 17aba2fb408d4f59bb99bba7ec324a0794aed2f7
