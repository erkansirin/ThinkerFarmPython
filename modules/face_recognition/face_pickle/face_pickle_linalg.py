# face_pickle.py
#
#
# Author by Erkan SIRIN
# Created for AI Edge project.
#
# face_pickle uses Python pickle module to serializing and de-serializing
# face object (The pickle module implements binary protocols for serializing
# and de-serializing a Python object structure.)

import tkinter as tki
import os
import cv2
import pickle
import imutils
from PIL import Image
from PIL import ImageTk
import numpy as np
import time
from db.db import db
from modules.nn_utility import *

class face_pickle_linalg:

    def __init__(self):

        print("started")

def load_pickle_face_network(self):

    self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
    self.pb.start()
    print("AI Edge : Loading Face Detector Network")
    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Loading Face Detector Network")

    protoPath = os.path.sep.join([self.root_path,"/models/face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join([self.root_path,"/models/face_detection_model",
        "res10_300x300_ssd_iter_140000.caffemodel"])
    torch_model = os.path.sep.join([self.root_path,"/models/face_detection_model","openface_nn4.small2.v1.t7"])

    self.net_utility = nn_utility(self.cpu_type)
    self.net_utility.setup_network(protoPath, modelPath,  "caffe")
    self.net_embedder = nn_utility(self.cpu_type)
    self.net_embedder.setup_network("", torch_model, "Torch")

    print("AI Edge : Loading Face Recognizer Network")
    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Loading Face Recognizer Network")

    self.recognizer = pickle.loads(open(os.path.sep.join([self.root_path,"/models/face_recognition_models/encodings.pickle"]), "rb").read())
    self.le = pickle.loads(open(os.path.sep.join([self.root_path,"/models/face_recognition_models/le.pickle"]), "rb").read())

    print("AI Edge : Starting Video Stream")
    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Starting Video Stream")

    self.active_menu = 0

def run_face_pickle(self,frame,image_blob):



    (h, w) = self.frame.shape[:2]

    detections = self.net_utility.nn_detector(image_blob)

    if self.update_final_text == 0:
        self.T.delete("1.0", tki.END)
        self.T.insert("1.0","System ready - AI Edge Face Module : detection with res10_300x300_ssd_iter_140000.caffemodel and Recognition with custom trained NN with  human dataset runing on OpenCV DNN")
        self.update_final_text = 1

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.1:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            print("detections box new :",box)
            print("detections.shape[2] new :",detections.shape[2])

            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            boxes = [(startY, endX, endY,startX)]
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
            encodings = face_encodings(rgb, boxes)

            if fW < 20 or fH < 20:
                continue

            # faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            # (96, 96), (0, 0, 0), swapRB=True, crop=False)
            #
            # vec = self.net_embedder.nn_detector(faceBlob)
            encodings = face_encodings(rgb, boxes)
            ts = time.time()

            #preds = self.recognizer.predict_proba(vec)[0]
            preds = self.recognizer["encodings"].predict_proba(encodings)
            j = np.argmax(preds)
            proba = preds[j]
            name = self.le.classes_[j]
            tstext= "%d" % ts

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
            humanid = "{}/dataset/say_300_.jpg".format(self.root_path)
            cv2.imwrite(humanid, frame)


            if proba > self.face_confidence:

                humanid = "humans/{}/peopled{}_conf_{:.2f}.jpg".format(name,tstext,proba * 100)
                text = "Staff ID : {} - {} ".format(name,db['people'][int(name)]['name'])

                self.top_label_text.set(text)
                img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/youcanpass.png"]))
                self.panel_pass.configure(image=img)
                self.panel_pass.image = img

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)

            else:
                humanid = "/home/pi/Desktop/ageofai/data/peopled{}_conf_{:.2f}.jpg".format(tstext,proba * 100)
                text = "Staff ID : Unknown"

                self.top_label_text.set(text)
                img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/whoareyou.png"]))
                self.panel_pass.configure(image=img)
                self.panel_pass.image = img

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)



    return frame