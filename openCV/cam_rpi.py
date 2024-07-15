#!/usr/bin/python3

# based on
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# and converted to Picamera2
# based on
# https://forums.raspberrypi.com/viewtopic.php?p=2214998#p2214998

import numpy as np
import cv2 as cv

from picamera2 import Picamera2

camera_id = 0
# camera_id = "/dev/video_cam_C920"

# setup pi cam
cam = Picamera2()
height = 480
width = 640
middle = (int(width / 2), int(height / 2))
cam.configure(
    cam.create_video_configuration(main={"format": "RGB888", "size": (width, height)})
)
cam.start()

while True:
    # Capture frame-by-frame
    frame = cam.capture_array()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow("frame", gray)
    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cam.stop()
cv.destroyAllWindows()
