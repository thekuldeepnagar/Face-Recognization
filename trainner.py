import cv2
import os
import numpy as np
from PIL import Image

path="dataset"

recognizer=cv2.face.LBPHFaceRecognizer_create()

detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getimages(path):

    imagepath=[os.path.join(path,f) for f in os.listdir(path)]
    facesample=[]
    ids=[]

    for imagepath in imagepath:
        pil_img=Image.open(imagepath).convert('L')
        img_numpy=np.array(pil_img,'uint8')

        id=int(os.path.split(imagepath)[-1].split(".")[1])
        faces=detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            facesample.append((img_numpy[y:y+h,x:x+w]))
            ids.append(id)

    return facesample,ids

faces,ids=getimages(path)
recognizer.train(faces,np.array(ids))

recognizer.write('train/trainer.yml')
