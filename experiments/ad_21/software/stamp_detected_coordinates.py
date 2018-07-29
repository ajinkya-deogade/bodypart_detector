#! /usr/bin/env python

from optparse import OptionParser
import json
import cv2
import numpy as np


def inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

if __name__ == '__main__':

    parser = OptionParser()
    # parser.add_option("", "--estimated-coordinate-list", dest="estimated_coordinates", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--estimated-coordinate-file", dest="estimated_coordinate_file", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--input-coordinate-file", dest="input_coordinate_file", default="",help="list of testing annotation JSON file")
    parser.add_option("", "--project-path", dest="project_dir", default="", help="path containing data directory")
    parser.add_option("", "--display", dest="display_level", default=0, type="int",help="display intermediate and final results visually, level 5 for all, level 1 for final, level 0 for none")
    # parser.add_option("", "--save-video-file", dest="save_dir_images", default="", help="directory to save result visualizations, if at all")
    parser.add_option("", "--video-file-one", dest="video_file_one", default="", help="Video file for first cropped image")
    parser.add_option("", "--video-file-two", dest="video_file_two", default="", help="Video file for second cropped image")
    parser.add_option("", "--save-video-file-one", dest="save_video_file_one", default="", help="path to store video alongwith its name ")
    parser.add_option("", "--save-video-file-two", dest="save_video_file_two", default="", help="path to store video alongwith its name ")

    (options, args) = parser.parse_args()

    bodypart_coords_est = {}
    frame_number = 0
    detected = 0
    correct_box = 0
    est_coordinates = []
    box_size = 200
    with open(options.estimated_coordinate_file) as fin_est_coordinates:
        for line in fin_est_coordinates:
            try:
                est_coordinate = json.loads(line.rstrip())
                est_coordinates.append(est_coordinate)
            except ValueError:
                print "Not Read the Line", line
                pass

    with open(options.input_coordinate_file) as fin_in_coordinates:
        input_coordinates = np.loadtxt(fin_in_coordinates, delimiter="\t", skiprows=1)

    video_file_1 = options.video_file_one
    print "Video File: ", video_file_1
    cap_1 = cv2.VideoCapture(video_file_1)

    video_file_2 = options.video_file_two
    print "Video File: ", video_file_2
    cap_2 = cv2.VideoCapture(video_file_2)

    resol_width_1 = int(cap_1.get(3))
    resol_height_1 = int(cap_1.get(4))
    fps_1 = int(10)
    fourcc_1 = int(cap_1.get(6))

    resol_width_2 = int(cap_2.get(3))
    resol_height_2 = int(cap_2.get(4))
    fps_2 = int(10)
    fourcc_2 = int(cap_2.get(6))

    print "Estimated Coordinates Shape: ", np.shape(est_coordinates)
    print "Input Coordinates Shape: ", np.shape(input_coordinates)

    for i in range(2000, len(est_coordinates)-1):
        cap_1.set(1, float(est_coordinates[i]["detections"][0]["frame_index"]))
        cap_2.set(1, float(est_coordinates[i]["detections"][0]["frame_index"]))
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()
        cv2.imshow("Frame_1", frame_1)
        cv2.imshow("Frame_2", frame_2)

        frame_number += 1
        detection_box_1 = {}
        detection_box_2 = {}
        detection_box_1["x"] = -1
        detection_box_1["y"] = -1
        detection_box_2["x"] = -1
        detection_box_2["y"] = -1

        if "coord_x" in est_coordinates[i]["detections"][0]:
            bodypart_coords_est["x"] = est_coordinates[i]["detections"][0]["coord_x"]
            bodypart_coords_est["y"] = est_coordinates[i]["detections"][0]["coord_y"]
            bodypart_coords_est["conf"] = est_coordinates[i]["detections"][0]["conf"]
            bodypart_coords_est["frame_index"] = est_coordinates[i]["detections"][0]["frame_index"]
            box_1_points = [(input_coordinates[bodypart_coords_est["frame_index"]][1], input_coordinates[bodypart_coords_est["frame_index"]][2]), (input_coordinates[bodypart_coords_est["frame_index"]][1]+box_size, input_coordinates[bodypart_coords_est["frame_index"]][2]), (input_coordinates[bodypart_coords_est["frame_index"]][1]+box_size, input_coordinates[bodypart_coords_est["frame_index"]][2]+box_size),(input_coordinates[bodypart_coords_est["frame_index"]][1], input_coordinates[bodypart_coords_est["frame_index"]][2]+box_size)]
            box_2_points = [(input_coordinates[bodypart_coords_est["frame_index"]][3], input_coordinates[bodypart_coords_est["frame_index"]][4]), (input_coordinates[bodypart_coords_est["frame_index"]][3]+box_size, input_coordinates[bodypart_coords_est["frame_index"]][4]), (input_coordinates[bodypart_coords_est["frame_index"]][3]+box_size, input_coordinates[bodypart_coords_est["frame_index"]][4]+box_size),(input_coordinates[bodypart_coords_est["frame_index"]][3], input_coordinates[bodypart_coords_est["frame_index"]][4]+box_size)]
            detected += 1

            if inside_polygon(bodypart_coords_est["x"], bodypart_coords_est["y"], box_1_points):
                detection_box_1["x"] = int(bodypart_coords_est["x"] - input_coordinates[bodypart_coords_est["frame_index"]][1])
                detection_box_1["y"] = int(bodypart_coords_est["y"] - input_coordinates[bodypart_coords_est["frame_index"]][2])
                display_detection = np.array(frame_1.copy(), np.float)
                print "Frame : %d  Inside Box One" %(est_coordinates[i]["detections"][0]["frame_index"])
                display_detection /= 255.0
                cv2.circle(display_detection, (detection_box_1["x"], detection_box_1["y"]), 4, (0, 255, 255), thickness=-1)
                cv2.imshow("Detected Frame", display_detection)
                cv2.waitKey(50)
                correct_box += 1

            elif inside_polygon(bodypart_coords_est["x"], bodypart_coords_est["y"], box_2_points):
                detection_box_2["x"] = int(bodypart_coords_est["x"] - input_coordinates[bodypart_coords_est["frame_index"]][3])
                detection_box_2["y"] = int(bodypart_coords_est["y"] - input_coordinates[bodypart_coords_est["frame_index"]][4])
                display_detection = np.array(frame_2.copy(), np.float)
                display_detection /= 255.0
                print "Frame : %d  Inside Box Two" %(est_coordinates[i]["detections"][0]["frame_index"])
                cv2.circle(display_detection, (detection_box_2["x"], detection_box_2["y"]), 4, (0, 255, 255), thickness=-1)
                cv2.imshow("Detected Frame", display_detection)
                cv2.waitKey(50)

            else:
                print "Frame : %d  Not Inside any Polygon" %(est_coordinates[i]["detections"][0]["frame_index"])
                cv2.waitKey(50)
        else:
            print "Frame : %d  No coordinates Detected" %(est_coordinates[i]["detections"][0]["frame_index"])

    print "Total Number of Frames : ", frame_number
    print "Percentage of Detection: %d/%d = %g"%(detected, frame_number, float(float(detected)*float(100)/float(frame_number)))
    print "Percentage of Correct Detection: %d/%d = %g"%(correct_box, detected, float(float(correct_box)*float(100)/float(detected)))

    cv2.destroyAllWindows()

