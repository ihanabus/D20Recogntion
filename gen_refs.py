#####
# opens camera and enables photo capture
# base structure for other scripts
# 
# camera feed and image capture
# source: https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
#
# Canny
# source: https://docs.opencv.org/4.x/da/d5c/tutorial_canny_detector.html
# If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge
# If a pixel gradient value is below the lower threshold, then it is rejected.
# If the pixel gradient is between the two thresholds, then it will be accepted only if it is connected to a pixel that is above the upper threshold.
#####

### imports
import cv2 as cv
import sys
import os
import numpy as np
import json

### variables
ref_path = './References'
cam_path = './Captures'
cur_path = ref_path
ct_file = 'canny.json'
ref_file = 'ref.json'

window_cam = 'Camera'
window_canny = 'Canny'
bar_lower = 'Lower'
bar_upper = 'Upper'

# usually 0 or 1 depending if system already has camera
camID = 1

canny_lt = canny_ut = 0
start_lt = start_ut = 0
clt = 'clt'
cut = 'cut'

### helper methods
def camera_usage():
    print()
    print("How to use camera")
    print("- [space] to capture frame")
    print("- [esc] to quit")
    print()

def load_cts(file):
    if not os.path.isfile(file):
        return
    global canny_lt, canny_ut, start_lt, start_ut
    with open(file, 'r') as fp:
        data = json.load(fp)
        if len(data):
            canny_lt = start_lt = data[clt]
            canny_ut = start_ut = data[cut]
            print("loaded previous values")
            print(f"[{canny_lt}, {canny_ut}]")
            print()

def open_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def edge_detection(frame):
    global canny_lt, canny_ut
    frame_gs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    canny_lt = cv.getTrackbarPos(bar_lower, window_canny)
    canny_ut = cv.getTrackbarPos(bar_upper, window_canny)
    frame_canny = cv.Canny(frame_gs, canny_lt, canny_ut)
    return frame_canny

def auto_crop(frame):
    frame_canny = edge_detection(frame)
    x, y, w, h = cv.boundingRect(frame_canny)
    roi_rgb = frame[y:y+h, x:x+w]
    roi_gs = frame_canny[y:y+h, x:x+w]
    return roi_rgb, roi_gs

### callback functions for trackbars
def null_callback(x):
    pass

### windows
cv.namedWindow(window_cam)
cv.namedWindow(window_canny)
### trackbars
cv.createTrackbar(bar_lower, window_canny, 0, 255, null_callback)
cv.createTrackbar(bar_upper, window_canny, 0, 255, null_callback)

### program start
cam = cv.VideoCapture(camID)   
if not cam.isOpened():
    print()
    print("Unable to access camera")
    print()
    sys.exit()

camera_usage()
open_dir(ref_path)
open_dir(cam_path)

load_cts(ct_file)
cv.setTrackbarPos(bar_lower, window_canny, canny_lt)
cv.setTrackbarPos(bar_upper, window_canny, canny_ut)

while True:

    # ret: boolean for successful frame capture
    # frame: HxWxC (C->BGR)
    ret, frame = cam.read()                 
    if not ret:
        print("Can't receive frame, exiting...")
        break
    cv.imshow(window_cam, frame)

    # Canny edge detection
    frame_canny = edge_detection(frame)
    cv.imshow(window_canny, frame_canny)
    
    # user input
    k = cv.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        print()
        break
    elif k % 256 == 32:
        # SPACE pressed
        crop_rgb, crop_gs = auto_crop(frame)
        if crop_rgb is None:
            continue
        elif 0 in crop_rgb.shape:
            continue
        else: 
            file_name = input("file name (.png): ")
            file_name += '.png'
            cv.imwrite(os.path.join(cur_path, file_name), crop_rgb)
            print(f"{file_name} {crop_rgb.shape} written!")
            print()

cam.release()

cv.destroyAllWindows()

# save new values
if canny_lt != start_lt or canny_ut != start_ut:
    print("Canny threshold on exit")
    print(f"[{canny_lt}, {canny_ut}]")
    print()
    input = input("SAVE above threshold? ([y] for yes): ")
    if input in ['y', '[y]', 'yes']: 
        canny_dict = {}
        canny_dict[clt] = canny_lt
        canny_dict[cut] = canny_ut
        with open(ct_file, 'w') as file:
            json.dump(canny_dict, file)    
        print("Saved new values")
        print()