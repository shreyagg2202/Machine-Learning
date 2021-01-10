# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:10:54 2021

@author: Shrey
"""

import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is():
        print("Face Not Found")
        return None
    
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+h]
        
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret,frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name_path = r'C:\Users\91701\OneDrive\Desktop\CETPA ML\ML Project Face Recognition\Train Img\user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0),3)
        cv2.imshow("Face Cropper",face)
        
    else:
        face_extractor(frame)
        pass
    
    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Completed")