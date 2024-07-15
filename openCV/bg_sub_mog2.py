#!/usr/bin/python3

# https://docs.opencv2.org/3.4/d1/dc5/tutorial_background_subtraction.html

from __future__ import print_function
import cv2
import argparse

camera_id = 0
# camera_id = "/dev/video_cam_C920"

parser = argparse.ArgumentParser(
    description="This program shows how to use background subtraction methods provided by \
    OpenCV2. You can process both videos and images."
)
parser.add_argument(
    "--input",
    type=str,
    help="Path to a video or a sequence of image.",
    default=camera_id,
)
parser.add_argument(
    "--algo",
    type=str,
    help="Background subtraction method (KNN, MOG2).",
    default="MOG2",
)
args = parser.parse_args()

if args.algo == "MOG2":
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()


capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))

if not capture.isOpened():
    print("Unable to open: " + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(
        frame,
        str(capture.get(cv2.CAP_PROP_POS_FRAMES)),
        (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
    )

    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)

    if cv2.waitKey(1) == ord("q"):
        break
