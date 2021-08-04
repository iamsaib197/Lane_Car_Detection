import cv2

from cv2 import CascadeClassifier

import numpy

import pandas as pd

cap = cv2.VideoCapture('C:/Users/HP/PycharmProjects/lane_Car_detection/yellow.mp4')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('cars.xml')

# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3, 1)

#for drawing rectangles over the cars
    for (x ,y ,w ,h) in cars:

        cv2.rectangle(frames ,(x ,y) ,( x +w , y +h) ,(0 ,0 ,255) ,2)

    # Display frames in a window
    cv2.imshow('video2', frames)

#If we wish to stop the detection mode
    if cv2.waitKey(33) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()