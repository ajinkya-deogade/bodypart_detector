#! /opt/local/bin/python

from optparse import OptionParser
import json
import os
import re
import cv2
import numpy as np
import time

def main(options, args):

    test_annotations = []

    with open(options.test_annotation_list) as fin_annotation_list:
        for test_annotation_file in fin_annotation_list:
            test_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", test_annotation_file.strip()))
            with open(test_annotation_file) as fin_annotation:
                test_annotation = json.load(fin_annotation)
                test_annotations.append(test_annotation)
                # test_FileNames.extend(test_annotation["VideoFile"])
                #test_annotations["VideoFile"].append(test_annotation["VideoFile"])
                #test_annotations["Annotations"].extend(test_annotation["Annotations"])
    print "len(test_annotations):", len(test_annotations)

    new_test_annotations = []
    bodypart_names = ["MouthHook","LeftMHhook","RightMHhook","LeftDorsalOrgan","RightDorsalOrgan"]

    for i in range(0, len(test_annotations)):
        frame_index = -1
        video_file = test_annotations[i]["VideoFile"]
        video_file = re.sub(".*/data/", "data/", video_file)
        video_file = os.path.join(options.project_dir, video_file)
        cap = cv2.VideoCapture(video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print "Video..........%02d"%(i)

        annotations = []
        new_annotations = {}
        new_annotations["Annotations"] = []
        annotations.extend(test_annotations[i]["Annotations"])
        annotatedFrame_VideoIndex = [int(a["FrameIndexVideo"]) for a in annotations]

        interval = annotatedFrame_VideoIndex[1] - annotatedFrame_VideoIndex[0]
        annotatedFrame_VideoIndex = np.array(annotatedFrame_VideoIndex)

        new_annotation_folder = os.path.dirname(video_file) + '/new_annotations/'
        if os.path.exists(new_annotation_folder) != 1:
            os.mkdir(new_annotation_folder)
        videoFolder = new_annotation_folder + os.path.basename(video_file)[:-4]
        if os.path.exists(videoFolder) != 1:
            os.mkdir(videoFolder)

        new_annotation_file = videoFolder + "/" + os.path.basename(video_file)[:-4] + "_Coordinates.json"
        fileWriter = open(new_annotation_file, 'w+')

        xValues_Frames = {}
        yValues_Frames = {}
        interpolated_xvals = {}
        interpolated_yvals = {}

        for bp in range(0, len(bodypart_names)):
            bpName = bodypart_names[bp]
            xValues_Frames[bp] = [a["FrameValueCoordinates"][bp]["Value"]["x_coordinate"] for a in annotations if bpName == a["FrameValueCoordinates"][bp]["Name"]]
            yValues_Frames[bp] = [a["FrameValueCoordinates"][bp]["Value"]["y_coordinate"] for a in annotations if bpName == a["FrameValueCoordinates"][bp]["Name"]]

            interpolated_xvals[bp] = {}
            interpolated_yvals[bp] = {}
            interpolated_xvals[bp][0] = xValues_Frames[bp][0]
            interpolated_yvals[bp][0] = yValues_Frames[bp][0]
            for fid in range(1, len(annotatedFrame_VideoIndex)):
                presentIndex = annotatedFrame_VideoIndex[fid]
                previousIndex = annotatedFrame_VideoIndex[fid-1]
                interpolated_xvals[bp][previousIndex] = xValues_Frames[bp][fid-1]
                interpolated_yvals[bp][previousIndex] = yValues_Frames[bp][fid-1]
                interpolated_xvals[bp][presentIndex] = xValues_Frames[bp][fid]
                interpolated_yvals[bp][presentIndex] = yValues_Frames[bp][fid]
                if xValues_Frames[bp][fid-1] != -1 and yValues_Frames[bp][fid-1] != -1 and xValues_Frames[bp][fid] != -1 and yValues_Frames[bp][fid] != -1:
                    fp = [previousIndex, presentIndex]
                    xp = [xValues_Frames[bp][fid-1], xValues_Frames[bp][fid]]
                    yp = [yValues_Frames[bp][fid-1], yValues_Frames[bp][fid]]
                    for val in range(previousIndex+1, presentIndex):
                        interpolated_xvals[bp][val] = np.interp(val, fp, xp)
                        interpolated_yvals[bp][val] = np.interp(val, fp, yp)
                else:
                    for val in range(previousIndex+1, presentIndex):
                        interpolated_xvals[bp][val] = -1
                        interpolated_yvals[bp][val] = -1

        print "Saving frames......."
        while True:
            presentAnnotation = {}
            frame_index += 1
            ret, frame = cap.read()
            fileName =  videoFolder + '/' + '%05d'%(frame_index+1) +  "_"  +  os.path.basename(video_file)[:-4] + '.tiff'
            cv2.imwrite(fileName, frame)
            if (options.display_level >= 2):
                display_voters = frame.copy()
                cv2.imshow("Frame", display_voters)
                cv2.waitKey(200)

            presentAnnotation["FrameID"] = frame_index+1
            presentAnnotation["FrameIndexVideo"] = frame_index+1
            presentAnnotation["FrameFile"] = fileName
            presentAnnotation["FrameValueCoordinates"] = []

            for bp in range(0, len(bodypart_names)):
                temp = {}
                temp["Name"] = bodypart_names[bp]
                temp["Value"] = {}
                if frame_index in interpolated_xvals[bp]:
                    temp["Value"]["x_coordinate"] = interpolated_xvals[bp][frame_index]
                    temp["Value"]["y_coordinate"] = interpolated_yvals[bp][frame_index]
                else:
                    temp["Value"]["x_coordinate"] = -1
                    temp["Value"]["y_coordinate"] = -1

                presentAnnotation["FrameValueCoordinates"].append(temp)

            new_annotations["Annotations"].append(presentAnnotation)
            if frame_index == length-1:
                break
        cap.release()
        cap = None

        print "Saving JSON file......"
        json.dump(new_annotations, fileWriter, sort_keys=True, indent=4, separators=(',', ': '))
        fileWriter.close()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("", "--test-annotation-list", dest="test_annotation_list", default="", help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")

    (options, args) = parser.parse_args()

    main(options, args)