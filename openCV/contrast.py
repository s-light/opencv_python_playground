#!/usr/bin/python3

# https://docs.opencv2.org/4.x/dd/d43/tutorial_py_video_display.html

import numpy as np
import cv2

camera_id = 0
# camera_id = "/dev/video_cam_C920"


frame_width = 700
frame_height_offset = 600

cv2.namedWindow("raw")
cv2.namedWindow("gray")
cv2.namedWindow("denoised")
cv2.namedWindow("threshold gaus")

cv2.moveWindow("raw", 0, 0)
cv2.moveWindow("gray", 1 * frame_width, 0)
cv2.moveWindow("denoised", 2 * frame_width, 0)
cv2.moveWindow("threshold gaus", 3 * frame_width, 0)


cv2.namedWindow("contrast Weighted")
cv2.namedWindow("contrast gimp")
cv2.namedWindow("contrast LAB CLAHE")
cv2.namedWindow("contrast normalize")

cv2.moveWindow("contrast Weighted", 0 * frame_width, frame_height_offset)
cv2.moveWindow("contrast gimp", 1 * frame_width, frame_height_offset)
cv2.moveWindow("contrast LAB CLAHE", 2 * frame_width, frame_height_offset)
cv2.moveWindow("contrast normalize", 3 * frame_width, frame_height_offset)


cv2.namedWindow("contrast CLAHE2")
cv2.moveWindow("contrast CLAHE2", 4 * frame_width, 2 * frame_height_offset)

# fullscreen
# cv.namedWindow("foo", cv.WINDOW_NORMAL)
# cv.setWindowProperty("foo", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


def adjust_contrast_brightness(input_img, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving as is
    brightness: [-255, 255] with 0 leaving as is

    written by: pietz
    source: https://stackoverflow.com/a/69884067/574981
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(input_img, contrast, input_img, 0, brightness)


def apply_brightness_contrast_gimp(input_img, brightness=0, contrast=0):
    """
    brightness: [0, 255]
    contrast: [-127, 127] 0 = no change
    written by: bfris
    source: https://stackoverflow.com/a/50053219/574981
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def adjust_contrast_CLAHE(input_img, contrast=1.0):
    """
    contrast: [0.0, inf] 1.0 = no change
    written by: Jeru Luke
    source: https://stackoverflow.com/a/41075028/574981
    """
    # converting to LAB color space
    lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    # result = np.hstack((input_img, enhanced_img))
    return enhanced_img


# create a CLAHE object (Arguments are optional).
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = cv2.createCLAHE()


def adjust_contrast_CLAHE2(input_img):
    """
    source: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    """
    return clahe.apply(input_img)


exit_key_list = [ord("q"), 27]

cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    cv2.imshow("raw", frame)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    # Display the resulting frame
    cv2.imshow("gray", gray)
    cv2.imshow("denoised", denoised)
    cv2.imshow("contrast Weighted", adjust_contrast_brightness(frame, 20.0, 0))
    cv2.imshow("contrast gimp", apply_brightness_contrast_gimp(frame, 0, 127))
    cv2.imshow("contrast LAB CLAHE", adjust_contrast_CLAHE(frame, 1.0))
    cv2.imshow(
        "contrast normalize", cv2.normalize(frame, frame, 0, 2 * 255, cv2.NORM_MINMAX)
    )
    cv2.imshow("contrast CLAHE2", adjust_contrast_CLAHE2(denoised))
    cv2.imshow(
        "threshold gaus",
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ),
    )
    if cv2.waitKey(1) in exit_key_list:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
