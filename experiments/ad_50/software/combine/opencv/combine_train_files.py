#! /usr/bin/env python

from optparse import OptionParser
import json
from pprint import pprint
import cv2
import os
import re
import numpy as np
import pickle
import random

class SaveClass:
    def __init__(self, votes = [], keypoints = [], descriptors = [], bodypart = None, hessianThreshold = None, nOctaves = None, nOctaveLayers = None):
        self.votes = votes
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.bodypart = bodypart
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers

def main(options, args):
    
    train_features = SaveClass()
    with open(options.train_feature_list) as fin_feature_list:
        for train_feature_file in fin_feature_list:
            train_feature_file = train_feature_file.strip()
            print "loading", train_feature_file
            with open(train_feature_file, 'rb') as fin_feature_list:
                features_temp = pickle.load(fin_feature_list)
                assert(len(features_temp.votes) == len(features_temp.descriptors))
                print "number of features:", len(features_temp.votes)

                if (train_features.bodypart == None):
                    train_features.bodypart = features_temp.bodypart
                else:
                    assert(train_features.bodypart == features_temp.bodypart)

                if (train_features.hessianThreshold == None):
                    train_features.hessianThreshold = features_temp.hessianThreshold
                else:
                    assert(train_features.hessianThreshold == features_temp.hessianThreshold)

                if (train_features.nOctaves == None):
                    train_features.nOctaves = features_temp.nOctaves
                else:
                    assert(train_features.nOctaves == features_temp.nOctaves)

                if (train_features.nOctaveLayers == None):
                    train_features.nOctaveLayers = features_temp.nOctaveLayers
                else:
                    assert(train_features.nOctaveLayers == features_temp.nOctaveLayers)

                train_features.votes.extend(np.array(features_temp.votes, dtype=np.float32))
                train_features.descriptors.extend(np.array(features_temp.descriptors, dtype=np.float32))

    train_features.keypoints = np.arange(len(train_features.descriptors), dtype=np.float32)

    print "bodypart:", train_features.bodypart
    print "total number of features:", len(train_features.votes)
    
    with open(options.save_file, 'wb') as fout_save:
        pickle.dump(train_features, fout_save)


if __name__ == '__main__':
    parser = OptionParser()
    # Read the options
    parser.add_option("", "--train-feature-list", dest="train_feature_list", default="",help="list of files containing training feature vectors")
    parser.add_option("", "--save-file", dest="save_file", default="",help="save file name of combined features")

    (options, args) = parser.parse_args()

    main(options, args)
