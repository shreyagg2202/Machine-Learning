# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

img = cv2.imread(r"C:\Users\91701\OneDrive\Desktop\img\hri.jpg")
cv2.imshow("Hritik Rosshan", img)

print(img.shape)
cv2.waitKey(0)

