# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:47:06 2021

@author: 91701
"""

import cv2
import numpy as np

dev = cv2.VideoCapture(0)
while True:
    ret,frame = dev.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lower_range = np.array([110,50,50])  #for blue color detection
    upper_range = np.array([130,255,255])
    
    mask = cv2.inRange(hsv,lower_range,upper_range)
    cv2.imshow("Show", mask)
    cv2.imshow("Show 1", frame)
    
    if cv2.waitKey(1)==13:
        break

dev.release()