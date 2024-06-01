import cv2 as cv

cams_test = 500
for i in range(0, cams_test):
    cap = cv.VideoCapture(i)
    test, frame = cap.read()
    if test:
        print("i : "+str(i)+" /// result: "+str(test))