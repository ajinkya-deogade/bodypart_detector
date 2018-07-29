#! /opt/local/bin/python

from optparse import OptionParser
import json
import re
import csv
import os

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
    headers = ['FrameNumber','MouthHook_x','MouthHook_y','LeftMHhook_x','LeftMHhook_y','RightMHhook_x','RightMHhook_y','LeftDorsalOrgan_x','LeftDorsalOrgan_y','RightDorsalOrgan_x','RightDorsalOrgan_y',
               'CenterBolwigOrgan_x', 'CenterBolwigOrgan_y', 'LeftBolwigOrgan_x', 'LeftBolwigOrgan_y', 'RightBolwigOrgan_x', 'RightBolwigOrgan_y']

    if (options.train_annotation_file != ""):
        with open(options.train_annotation_file) as fin_annotation:
            train_annotation = json.load(fin_annotation)
    else:
        with open(options.train_annotation_list) as fin_annotation_list:
            for train_annotation_file in fin_annotation_list:
                train_annotation_file = os.path.join(options.project_dir,re.sub(".*/data/", "data/", train_annotation_file.strip()))
                with open(train_annotation_file) as fin_annotation:
                    train_annotation = []
                    save_folder = options.save_dir
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_name = os.path.join(save_folder, os.path.splitext(os.path.basename(train_annotation_file))[0]) + ".csv"
                    writer = csv.DictWriter(open(save_name, 'w'), delimiter=',',lineterminator='\n', fieldnames=headers)
                    # writer.writerows(headers)
                    tmp_train_annotation = json.load(fin_annotation)
                    for i in range(0, len(tmp_train_annotation["Annotations"])):
                        temp_row = {}
                        for m in headers:
                            temp_row[m] = None
                        for j in range(0, len(tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"])):
                            temp_row['FrameNumber']=int(tmp_train_annotation["Annotations"][i]["FrameIndexVideo"])
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "MouthHook"):
                                temp_row['MouthHook_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['MouthHook_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "LeftMHhook"):
                                temp_row['LeftMHhook_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['LeftMHhook_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightMHhook"):
                                temp_row['RightMHhook_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['RightMHhook_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "LeftDorsalOrgan"):
                                temp_row['LeftDorsalOrgan_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['LeftDorsalOrgan_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightDorsalOrgan"):
                                temp_row['RightDorsalOrgan_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['RightDorsalOrgan_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "CenterBolwigOrgan"):
                                temp_row['CenterBolwigOrgan_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['CenterBolwigOrgan_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "LeftBolwigOrgan"):
                                temp_row['LeftBolwigOrgan_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['LeftBolwigOrgan_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                            if (tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Name"] == "RightBolwigOrgan"):
                                temp_row['RightBolwigOrgan_x']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["x_coordinate"]
                                temp_row['RightBolwigOrgan_y']=tmp_train_annotation["Annotations"][i]["FrameValueCoordinates"][j]["Value"]["y_coordinate"]
                        train_annotation.append(temp_row)
                    # print "Train Annotation: ", train_annotation
                    writer.writerows(train_annotation)
