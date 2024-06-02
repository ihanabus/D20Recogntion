#####
# opens camera and enables photo capture
# base structure for other scripts
# 
# camera feed and image capture
# source: https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
#####

import cv2 as cv
import os
import numpy as np

print("OpenCV version:", cv.__version__)
print()

img_path = './Images'

# usually 0 or 1 depending if system already has camera
camID = 1

cam = cv.VideoCapture(camID)    

if not cam.isOpened():
    print("Unable to access camera")
    print()
    exit()

# camera use
print("Instructions")
print("- [space] to capture frame")
print("- [esc] to quit")
print(f"save directory: {img_path}")
print()

window_cam = 'Camera'
# window_gray = 'Grayscale'
cv.namedWindow(window_cam)
# cv.namedWindow(window_gray)

while True:

    # ret: boolean for successful frame capture
    # frame: HxWxC (C->BGR)
    ret, frame = cam.read()                 
    if not ret:
        print("Can't receive frame, exiting...")
        break

    cv.imshow(window_cam, frame)

    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow(window_gray, frame_gray)
    
    # wait in ms for user input
    # keep low, otherwise frame rate decreases
    k = cv.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        print()
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame.png"
        cv.imwrite(os.path.join(img_path, img_name), frame)
        print(f"{img_name} {frame.shape} written!")
        print()

cam.release()

cv.destroyAllWindows()