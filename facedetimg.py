
import cv2
import numpy as np
i=0
#img = cv2.imread('download1.jpg')
img = cv2.imread('Group2016.jpg')
#img = cv2.imread('cars.jpg')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('car2.xml')
faces = face_cascade.detectMultiScale(img,1.3,5)
for(x,y,w,h) in faces:
        sub_face = img[y:y+h, x:x+w]
        file = "face"+str(i)+".jpg"
        i=i+1
        cv2.imwrite(file, sub_face)
        #cv2.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
        

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
