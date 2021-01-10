# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:41:34 2021

@author: 91701
"""

import cv2
import numpy as np

face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye=cv2.CascadeClassifier("frontalEyes35x16.xml")

image=cv2.imread("hri.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

face=face.detectMultiScale(gray,1.3,5)

if face is():
    print("Face")
    
for (x,y,w,h) in face:
    image=cv2.rectangle(image,(x,y),(x+w,y+w),(127,0,2025),3)
cv2.imshow("Face Detected",image)
cv2.waitKey(0)
cv2.destroyAllWindows()