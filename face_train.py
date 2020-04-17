import os
import numpy as np
from PIL import Image
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

current_id = 0
label_ids = {}
x_train = []
y_lable = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png"):
            path = os.path.join(root,file)
            lable = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            
            if not lable in label_ids:
                label_ids[lable] = current_id
                current_id += 1
            id_ = label_ids[lable]
            
            
            pil_image = Image.open(path).convert("L")
            gray = np.array(pil_image,"uint8")
            faces = face_cascade.detectMultiScale(gray)            
            for (x,y,w,h) in faces:
                roi = gray[y:y+h, x:x+w]
                x_train.append(roi)
                y_lable.append(id_)

recognizer = cv2.face.LBPHFaceRecognizer_create()
with open("label.pickle", "wb") as f:
    pickle.dump(label_ids, f)
 
recognizer.train(x_train, np.array(y_lable))
recognizer.save("trainner.yml")