import cv2
from datetime import datetime
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
cap = cv2.VideoCapture(0)
labels = {}
with open("label.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
while (True) : 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=30:
            print(id_)
            print(labels)
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            color = (0,255,0)
            stroke = 2
            cv2.putText(frame, name, (x,y), font,1,color,stroke,cv2.LINE_AA)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        color = (0, 255, 0)
        stroke = 2
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
        
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q') : 
        break

cap.release()
cv2.destroyAllWindows()