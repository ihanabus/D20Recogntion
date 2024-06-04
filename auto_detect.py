#####
# opens camera with automated capture
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
#
# 
# https://github.com/Artefact2/autodice
# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
#
#####

### imports
import cv2 as cv
import sys
import os
import numpy as np
import math
import json
import glob
import time

### socket
import sys
from socket import socket, AF_INET, SOCK_DGRAM

SERVER_IP = b'100.81.37.36'
PORT_NUMBER = 5000
SIZE = 1024

sock = socket(AF_INET, SOCK_DGRAM)

### variables
ref_path = './References'
cam_path = './Captures'
cur_path = cam_path
ct_file = 'canny.json'

window_cam = 'Camera'
window_canny = 'Canny'
bar_lower = 'Lower'
bar_upper = 'Upper'

# usually 0 or 1 depending if system already has camera
camID = 2

canny_lt = canny_ut = 0
start_lt = start_ut = 0
clt = 'clt'
cut = 'cut'

ref_data = dict()

prev_number = None
interval = 1        # second interval of captures

###
sift = cv.SIFT.create(nOctaveLayers = 1)
bf = cv.BFMatcher()
dist_ratio = 0.8

# returns key points and descriptors of given image
def keypointsAndDescriptors(img):
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

# brute force matching of descriptors with k-nearest neighbors approach
def matched(des1, des2):
    if des1 is None or des2 is None:
        return None
    matches = bf.knnMatch(des1, des2, k = 2)
    good_matches = []
    # apply ratio test for stronger matches
    if len(matches[0]) != 2:
        return None
    for m,n in matches:
        if m.distance < dist_ratio * n.distance:
            good_matches.append((m,n))
    # [(m, n) for (m, n) in matches if m.distance < .8 * n.distance]
    return good_matches

def findHomographyAndInliers(kp1, kp2, matches):
    # Note:
    # queryIdx and trainIdx are naming conventions
    # in the case of comparing two images, these names don't mean much
    srcPoints = np.float32([ kp1[m.queryIdx].pt for (m, n) in matches ]).reshape(-1, 1, 2)
    dstPoints = np.float32([ kp2[m.trainIdx].pt for (m, n) in matches ]).reshape(-1, 1, 2)

    # find perspective transformation matrix M (mask indicates outlier status)
    M, mask = cv.findHomography(srcPoints, dstPoints, cv.RANSAC, 10.0)
    if mask is None:
        return [], [], 0
    
    matchesMask = [ [k,0] for k in mask.ravel().tolist()]
    return M, matchesMask, np.count_nonzero(mask)

def distanceToCenter(x, y, w, h):
    dx = x - .5 * (w-1)
    dy = y - .5 * (h-1)
    return math.sqrt(dx*dx + dy*dy)

def scoreMatches(matches, mask, kp1, w1, h1, kp2, w2, h2):
    score = 0.0
    
    for i, (m, n) in enumerate(matches):
        if not mask[i][0]:
            continue
        
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        r1 = distanceToCenter(pt1[0], pt1[1], w1, h1)
        r2 = distanceToCenter(pt2[0], pt2[1], w2, h2)

        score += 1.0 / (1.0 + max(0, abs(r1 / 30.0) - 1)) / (1.0 + max(0, abs((r2-r1) / 10.0) - 1))
        
    return score

def scoreCenterDiff(img1, img2, p):
    h, w, d = img1.shape
    y1 = int((.5 - p / 2.0) * h)
    y2 = int((.5 + p / 2.0) * h)
    x1 = int((.5 - p / 2.0) * w)
    x2 = int((.5 + p / 2.0) * w)

    s1 = img1[y1:y2,x1:x2].view(dtype=np.int8)
    s2 = img2[y1:y2,x1:x2].view(dtype=np.int8)
    
    return np.linalg.norm(s2 - s1) / ((y2 - y1)*(x2 - x1))

def scoreFinal(sMatch, sCenterDiff):
    return sMatch - 10.0 * sCenterDiff

def refmatch(img, ref_data):
    kp, des = keypointsAndDescriptors(img)
    
    inl = []
    for i in ref_data:

        # ignore if number of matches is too low
        matches = matched(ref_data[i][0][1], des)
        if matches is None:
            continue
        if len(matches) <= 10:
            continue
        
        # ignore if number of point pairs that fall below
        # the reprojection error threshold is too low
        M, matchesMask, nInliers = findHomographyAndInliers(ref_data[i][0][0], kp, matches)
        if nInliers <= 10:
            continue

        h1, w1, d = ref_data[i][1].shape
        h2, w2, d = img.shape

        # align perspective of reference image with input image
        warped = cv.warpPerspective(ref_data[i][1], M, (w2, h2))

        # score and penalize
        sMatches = scoreMatches(matches, matchesMask, ref_data[i][0][0], w1, h1, kp, w2, h2)
        sCenterDiff = scoreCenterDiff(img, warped, .333)
        
        inl.append((scoreFinal(sMatches, sCenterDiff), i))

    inl = sorted(inl, reverse = True)

    if len(inl) > 0:
        # print(inl[0][0])
        return inl[0][1]
    else:
        return None

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

def load_ref(ref_path):
    global ref_data
    for file in glob.glob(f'{ref_path}/*.png'):
        i = file.replace(ref_path, '')[1:].split('.')[0]
        i = int(i)
        img = cv.imread(file)
        ref_data[i] = (
            keypointsAndDescriptors(img),
            img,
        )
    print(f"Loaded {len(ref_data)} reference images")
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

# tries to pick up on new numbers on a timer
# includes transitions between numbers to None
def crop_and_update(frame):
    global prev_number
    if frame is None:
        return
    result = None
    crop_rgb, crop_gs = auto_crop(frame)
    if crop_rgb is not None:
        h,w,_ = crop_rgb.shape
        # minimum ROI area (above 100x100xc if square)
        if h*w > 10000:
            result = refmatch(crop_rgb, ref_data)
    if prev_number is None:
        if result is not None:
            img_name_rgb = "crop_rgb.png"
            path = os.path.join(cur_path, img_name_rgb)
            cv.imwrite(path, crop_rgb)
            prev_number = result
            obj = {}
            obj['d20_result'] = result
            deliverable = json.dumps(obj).encode('utf-8')
            sock.sendto(deliverable, (SERVER_IP, PORT_NUMBER))
            print(result)
    else:
        if result is None:
            prev_number = None
            

### callback functions for trackbars
def null_callback(x):
    pass

### windows
cv.namedWindow(window_cam, cv.WINDOW_NORMAL)
cv.namedWindow(window_canny, cv.WINDOW_NORMAL)
# resize
cv.resizeWindow(window_cam, (500,500))
cv.resizeWindow(window_canny, (500, 500))
### trackbars
cv.createTrackbar(bar_lower, window_canny, 0, 255, null_callback)
cv.createTrackbar(bar_upper, window_canny, 0, 255, null_callback)

### program start
cam = cv.VideoCapture(camID)  
cam.set(cv.CAP_PROP_FPS, 30)

if not cam.isOpened():
    print()
    print("Unable to access camera")
    print()
    sys.exit()

camera_usage()
open_dir(ref_path)
open_dir(cam_path)

load_cts(ct_file)
load_ref(ref_path)
cv.setTrackbarPos(bar_lower, window_canny, canny_lt)
cv.setTrackbarPos(bar_upper, window_canny, canny_ut)

start_time = time.monotonic()

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
    
    
    now_time = time.monotonic()
    # print((now_time - start_time) % interval)
    if (now_time - start_time) % interval < 0.05:
        crop_and_update(frame)    

    # user input
    k = cv.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        print() 
        break

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