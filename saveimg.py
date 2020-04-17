import cv2
from datetime import datetime

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while (True): 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        cv2.imwrite(current_time+".png",roi_gray)
        
        color = (0, 255, 0)
        stroke = 2
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q') : 
        break