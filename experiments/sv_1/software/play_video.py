#! /usr/bin/env python

from optparse import OptionParser

import cv2

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="video_file", default="", help="video file to play")

    (options, args) = parser.parse_args()

    print "video_file:" , options.video_file

    video_in = cv2.VideoCapture(options.video_file)
    fps = video_in.get(cv2.cv.CV_CAP_PROP_FPS)
    print "fps:" , fps

    frame_dur = int(1000.0 / float(fps))
    print "frame duration (ms):" , frame_dur

    while (True):
        ret, frame = video_in.read()

        print ret

        if (ret == False):
            break

        cv2.imshow(options.video_file, frame)

        k = cv2.waitKey(frame_dur)

    video_in.release()
