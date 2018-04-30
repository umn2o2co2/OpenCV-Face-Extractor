import cv2
import numpy as np
i=0
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('Marvel Studios Avengers Infinity War Official Trailer.mp4')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        sub_face = img[y:y+h, x:x+w]
        file = "face"+str(i)+".jpg"
        i=i+1
        cv2.imwrite(file, sub_face)
        for(ex,ey,ew,eh)in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        #smile = smile_cascade.detectMultiScale(roi_gray)
        #for(ex,ey,ew,eh)in smile:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break;

cap.release()
cv2.destroyAllWindows()

