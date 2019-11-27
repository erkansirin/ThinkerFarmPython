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
import threading
import datetime
import requests,json
import base64



class face_dlib_twocam:

    def __init__(self):

        print("started")

def load_dlib_face_network_twocam(self):


    # now = datetime.datetime.now()
    #
    # headers = {'Content-Type': 'application/json'}
    # header = {"Content-type": "application/json",
    #           "accept": "*/*"}
    #
    #           #dd-MM-yyyy HH:mm:ss
    # date_time = now.strftime("%d-%m-%Y %H:%M:%S")
    # print("date and time:",date_time)
    # postdata = { 'cameraId': '1', 'image': '', 'date':date_time }
    #
    # print("post data : ",postdata)
    #
    # r = requests.post(url = 'http://139.162.142.162:1994/api/companies/1/employees/23/passing', data = json.dumps(postdata), headers=header)
    # pastebin_url = r.text
    # print("The pastebin URL is : ",pastebin_url)

    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Loading Dlib Face Network")

    print("AI Edge : AI Edge : Loading Dlib Face Network")

    dlib.DLIB_USE_CUDA

    self.dataEncodings = pickle.loads(open(os.path.sep.join([self.root_path,"/models/face_recognition_models","encodings.pickle"]), "rb").read())
    protoPath = os.path.sep.join([self.root_path,"/models/face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join([self.root_path,"/models/face_detection_model",
        "res10_300x300_ssd_iter_140000.caffemodel"])


    self.net_utility = nn_utility(self.cpu_type)
    self.net_utility.setup_network(protoPath, modelPath,  "caffe")


    self.active_menu = 16
    print("AI Edge : Running Face Dlib")


def run_face_dlib_twocam(self,framelocal):

    run_cnn_dlib_local(self,framelocal)


    #run_cnn_dlib_local(self,framelocal)

    # alpha = 1.0 # Simple contrast control
    # beta = 50    # Simple brightness control
    #
    #
    # for y in range(frameip.shape[0]):
    #     for x in range(frameip.shape[1]):
    #         for c in range(frameip.shape[2]):
    #             frameip[y,x,c] = np.clip(alpha*frameip[y,x,c] + beta, 0, 255)



    # imageBlob_ip = cv2.dnn.blobFromImage(
    # cv2.resize(frameip, (300, 300)), 1.0, (300, 300),
    # (104.0, 177.0, 123.0), swapRB=False, crop=False)






    # imageBlob_local = cv2.dnn.blobFromImage(
    # cv2.resize(framelocal, (300, 300)), 1.0, (300, 300),
    # (104.0, 177.0, 123.0), swapRB=False, crop=False)
    #
    #
    # # detections_ip = self.net_utility.nn_detector(imageBlob_ip)
    # # for i in range(0, detections_ip.shape[2]):
    # #     confidence = detections_ip[0, 0, i, 2]
    # #     if confidence > 0.2:
    # #         #self.thread_recognition_ip = threading.Thread(target=run_cnn_dlib, args=(self,frameip))
    # #         #self.thread_recognition_ip.start()
    # #
    # #         run_cnn_dlib(self,frameip)
    #
    #
    # detections_local = self.net_utility.nn_detector(imageBlob_local)
    # for i in range(0, detections_local.shape[2]):
    #     confidence_local = detections_local[0, 0, i, 2]
    #     if confidence_local > 0.2:
    #         run_cnn_dlib_local(self,framelocal)

            #self.thread_recognition_local = threading.Thread(target=run_cnn_dlib_local, args=(self,frameip))
            #self.thread_recognition_local.start()










def run_cnn_dlib(self,frame):

    frame_ip = imutils.resize(frame, width=525,height=480)


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgb = frame_ip[:, :, ::-1]
    #print("imageBlob_ip rgb: ",rgb)

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
            text = "Staf_ID_Unknown_ipcam"
            ts = time.time()
            tstext= "%d" % ts
            humanid = "{}/dataset/unknown/{}_{}_.jpg".format(self.root_path,text,tstext)
            cv2.imwrite(humanid, frame)
        else:
            ts = time.time()
            tstext= "%d" % ts
            text = "Staff_ID_{}_{}_ipcam".format(name,db['people'][int(name)]['name'])
            humanid = "{}/dataset/known/{}_{}_.jpg".format(self.root_path,text,tstext)
            cv2.imwrite(humanid, frame)

            now = datetime.datetime.now()
            date_time = now.strftime("%d-%m-%Y %H:%M:%S")

            headers = {'Content-Type': 'application/json'}
            header = {"Content-type": "application/json",
                      "accept": "*/*"}
            postdata = { 'cameraId': '1', 'image': '', 'date':date_time }

            r = requests.post(url = 'http://139.162.142.162:1994/api/companies/1/employees/%s/passing'%name, data = json.dumps(postdata), headers=header)
            pastebin_url = r.text
            print("The pastebin URL is : ",pastebin_url)


        print("found person on ip cam: ",text)
        print("found person : ",text)
        self.top_label_text.set(text)
        img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/youcanpass.png"]))
        self.panel_pass.configure(image=img)
        self.panel_pass.image = img

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)




def run_cnn_dlib_local(self,frame):




    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #rgb = frame_ip[:, :, ::-1]

    boxes = face_locations(rgb, model="cnn")
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
            text = "Staff_ID_Unknown_local_cam"
            ts = time.time()
            tstext= "%d" % ts
            humanid = "{}/dataset/unknown/{}_{}_.jpg".format(self.root_path,text,tstext)
            cv2.imwrite(humanid, frame)
        else:
            ts = time.time()
            tstext= "%d" % ts
            text = "Staff_ID_{}_local_cam".format(name)
            humanid = "{}/dataset/known/{}_{}_.jpg".format(self.root_path,text,tstext)
            cv2.imwrite(humanid, frame)

            now = datetime.datetime.now()
            date_time = now.strftime("%d-%m-%Y %H:%M:%S")

            headers = {'Content-Type': 'application/json'}
            header = {"Content-type": "application/json",
                      "accept": "*/*"}
            postdata = { 'cameraId': '2', 'image': '', 'date':date_time }

            r = requests.post(url = 'http://139.162.142.162:1994/api/companies/1/employees/%s/passing'%name, data = json.dumps(postdata), headers=header)
            pastebin_url = r.text
            print("The pastebin URL is : ",pastebin_url)


        print("found person local: ",text)
        self.top_label_text.set(text)
        img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/youcanpass.png"]))
        self.panel_pass.configure(image=img)
        self.panel_pass.image = img

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
