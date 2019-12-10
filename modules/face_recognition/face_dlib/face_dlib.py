# face_pickle.py
#
#
# Author by Erkan SIRIN
# Created for ThinkerFarm project.
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
from db.person_db import person_db
from modules.face_recognition.face_dlib.face_dlib_utility import *
import dlib
import datetime
import requests,json
import base64
from utilities.paths import dlib_encodings


class fr_with_dlib:

    def __init__(self):

        print("started")

def load_dlib_face_network(self):

    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","ThinkerFarm : Loading Dlib Face Network")

    print("ThinkerFarm : ThinkerFarm : Loading Dlib Face Network")

    dlib.DLIB_USE_CUDA = True

    self.dataEncodings = pickle.loads(open(os.path.sep.join([dlib_encodings]), "rb").read())
    self.active_menu = 2
    print("ThinkerFarm : Running Face Dlib")



def run_face_dlib(self,frame):

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)


    boxes = face_locations(rgb, model="hog")

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
            ts = time.time()
            tstext= "%d" % ts
            humanid = "{}/dataset/unknown/{}_{}_.jpg".format(self.root_path,text,tstext)
            print(humanid)
            cv2.imwrite(humanid, frame)
        else:
            text = "Staff_ID_{}_local_cam".format(name)

            ts = time.time()
            tstext= "%d" % ts
            humanid = "{}/dataset/known/{}_{}_.jpg".format(self.root_path,text,tstext)
            print(humanid)
            cv2.imwrite(humanid, frame)

            now = datetime.datetime.now()
            date_time = now.strftime("%d-%m-%Y %H:%M:%S")

            headers = {'Content-Type': 'application/json'}
            header = {"Content-type": "application/json",
                      "accept": "*/*"}
            postdata = { 'cameraId': '2', 'image': '', 'date':date_time }



        print("found person : ",text)
        self.top_label_text.set(text)
        img = ImageTk.PhotoImage(Image.open("ui/images/youcanpass.png"))
        self.panel_pass.configure(image=img)
        self.panel_pass.image = img

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return frame

def run_face_dlib_backend(frame):
    print("running dlib")

    dataEncodings = pickle.loads(open("models/face_recognition_models","encodings.pickle", "rb").read())

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
