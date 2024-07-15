#!/usr/bin/python3

import argparse
import cv2


print(__file__)
print("source of this example:")
print(
    "https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/"
)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
args = vars(ap.parse_args())

# camera_id = "/dev/video_cam_C920"
camera_id = 0
window_name = "OpenCV Barcode"

frame_width = 500

cv2.namedWindow("Security Feed")
cv2.namedWindow("Thresh")
cv2.namedWindow("Frame Delta")
cv2.moveWindow("Security Feed", 0, 0)
cv2.moveWindow("Thresh", 1 * (frame_width + 10), 0)
cv2.moveWindow("Frame Delta", 2 * (frame_width + 10), 0)


# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# initialize the first frame in the video stream
frame_avg_bg = None

# def resize_frame(frame, width):
#     # Get the current frame size
#     h, w, _ = frame.shape

#     # Resize the frame
#     scale_percent = 100 / (w / width_)
#     new_width = int(w * scale_percent / 100)
#     new_height = int(h * scale_percent / 100)
#     dim = (new_width, new_height)
#     return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def resize_frame(image, width=None, height=None, inter=cv2.INTER_AREA):
    # https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L65C1-L94C19
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def grab_contours(cnts):
    # https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L154C1-L175C16
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(
            (
                "Contours tuple must have length 2 or 3, "
                "otherwise OpenCV changed their cv2.findContours return "
                "signature yet again. Refer to OpenCV's documentation "
                "in that case"
            )
        )

    # return the actual contours array
    return cnts


# loop over the frames of the video
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    text = "Unoccupied"

    # resize
    frame = resize_frame(frame, width=frame_width)

    # convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # and blur it
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if frame_avg_bg is None:
        # frame_avg_bg = gray
        frame_avg_bg = gray.copy().astype("float")
        continue

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, frame_avg_bg, 0.2)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(frame_avg_bg))

    # compute the absolute difference between the current frame and frame_avg_bg
    thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

        # draw the text and timestamp on the frame
    cv2.putText(
        frame,
        "Room Status: {}".format(text),
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
