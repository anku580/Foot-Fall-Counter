# -*- coding: utf-8 -*-
# ########## Customer Analysis system to obtain information about customers who enter Chumbak stores
# and send to Google Analytics ###############
# ####################Swaroop Sudhanva Belur#####################
import time
import os
from datetime import datetime

from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import imutils

from conf import *
from process_images import make_sure_path_exists

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TO_PROCESS_FILE_PATH = os.path.join(BASE_PATH, "images_to_process.txt")


def always_true():
    """This is the function that will keep running to check for obstructions and then spawns a new thread to
       analyse a picture
    """
    previous_frame = None
    while True:
        print datetime.now()
        camera.capture(raw_capture, format="bgr")
        original_frame = raw_capture.array

        resized_frame = imutils.resize(original_frame, width=500)  # maybe replace this with PIL
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if previous_frame is None:
            previous_frame = gray
            raw_capture.truncate(0)
            continue

        frame_delta = cv2.absdiff(previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        for c in cnts:
            if cv2.contourArea(c) > MIN_MOTION_AREA:
                time_now = datetime.now()
                time_str = time_now.strftime("%H-%M-%S-%f")
                date_str = time_now.strftime("%d-%m-%Y")
                file_path = os.path.join(BASE_PATH, 'Photos', date_str, '{}.jpg'.format(time_str))
                print file_path
                make_sure_path_exists(os.path.split(file_path)[0])
                cv2.imwrite(file_path, original_frame)

                f = open(TO_PROCESS_FILE_PATH, "a")
                f.write(file_path + "\n")
                f.close()

                break

        previous_frame = gray
        raw_capture.truncate(0)


if __name__ == "__main__":
    # Initialising Camera
    camera = PiCamera()
    camera.exposure_mode = u'sports'
    camera.resolution = (MAIN_IMAGE_DIMENSIONS['width'], MAIN_IMAGE_DIMENSIONS['height'])
    camera.framerate = 2
    camera.rotation = 0
    camera.hflip = FLIP_IMAGE
    camera.vflip = FLIP_IMAGE
    time.sleep(2)

    raw_capture = PiRGBArray(camera, size=(MAIN_IMAGE_DIMENSIONS['width'], MAIN_IMAGE_DIMENSIONS['height']))
    time.sleep(0.1)

    always_true()

