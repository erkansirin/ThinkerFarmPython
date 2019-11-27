# face_pickle.py
#
#
# Author by Erkan SIRIN
# Created for AI Edge project.
#
#  face_dlib using dlib face recognition module to recognize faces

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
from modules.face_recognition.face_dlib.face_dlib_utility import *
import dlib


class fr_with_dlib_pickle:

    def __init__(self):

        print("started")

def load_dlib_pickle_face_network(self):

    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Loading Dlib Face Network")

    print("AI Edge : AI Edge : Loading Dlib Face Network")

    dlib.DLIB_USE_CUDA = True

    self.dataEncodings = pickle.loads(open(os.path.sep.join([self.root_path,"/models/face_recognition_models","encodings.pickle"]), "rb").read())

    protoPath = os.path.sep.join([self.root_path,"/models/face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join([self.root_path,"/models/face_detection_model",
        "res10_300x300_ssd_iter_140000.caffemodel"])


    self.net_utility = nn_utility(self.cpu_type)
    self.net_utility.setup_network(protoPath, modelPath,  "caffe")




    self.active_menu = 15
    print("AI Edge : Running Face Dlib")



def run_face_pickle_dlib(self,frame,image_blob):

    (h, w) = frame.shape[:2]
    #(h, w) = (480, 525)
    print("(h, w) :",(h, w))

    detections = self.net_utility.nn_detector(image_blob)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print("detections box:",(startX, startY, endX, endY) )



            boxes = [(startX, startY, endX, endY)]

            #boxes = face_locations(rgb, model="cnn")
            #print("boxes :",boxes)
            encodings = face_encodings(rgb, boxes)

            names = []

            for encoding in encodings:

                matches = compare_faces(self.dataEncodings["encodings"],
                    encoding)

                name = "Unknown"

                if True in matches:

                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = self.dataEncodings["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    name = max(counts, key=counts.get)

                names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):

                text = ""
                if name == "Unknown":
                    text = "Staff ID : Unknown"
                else:
                    text = "Staff ID : {} - {} ".format(name,db['people'][int(name)]['name'])


                print("found person : ",text)
                self.top_label_text.set(text)
                # img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/youcanpass.png"]))
                # self.panel_pass.configure(image=img)
                # self.panel_pass.image = img

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        #for i in range(0, detections.shape[2]):











    return frame

def run_face_dlib_backend(frame):
    print("running dlib")

    root_path = ""

    dataEncodings = pickle.loads(open(os.path.sep.join([root_path,"/models/face_recognition_models","encodings.pickle"]), "rb").read())

    (h, w) = frame.shape[:2]


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_locations(rgb, model="hog")
    encodings = face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:

        matches = compare_faces(dataEncodings["encodings"],
            encoding)
        name = "Unknown"

        if True in matches:

            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = dataEncodings["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

    return name
