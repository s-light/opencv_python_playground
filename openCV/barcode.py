#!/usr/bin/python3
# https://note.nkmk.me/en/python-opencv-barcode/

import cv2

camera_id = 0
# camera_id = "/dev/video_cam_C920"

window_name = "OpenCV Barcode"

bd = cv2.barcode.BarcodeDetector()
cap = cv2.VideoCapture(camera_id)

last_detected_code = ""

def decode_frame(frame):
    global last_detected_code
    ret_bc, decoded_info, _, points = bd.detectAndDecode(frame)
    if ret_bc:
        frame = cv2.polylines(frame, points.astype(int), True, (0, 255, 0), 3)
        for s, p in zip(decoded_info, points):
            if s:
                if last_detected_code != s:
                    last_detected_code = s
                    print(s)
                frame = cv2.putText(
                    frame,
                    s,
                    p[1].astype(int),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 200, 0),
                    2,
                    cv2.LINE_AA,
                )
    cv2.imshow(window_name, frame)

while True:
    ret, frame = cap.read()
    if ret:
        decode_frame(frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyWindow(window_name)
