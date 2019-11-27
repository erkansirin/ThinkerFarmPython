# face_scan.py
#
#
# Author by Erkan SIRIN
# Created for AI Edge project.
#
# face_scan scan faces and record images to human dataset folder

import tkinter as tki
import os
import cv2
import pickle
import imutils
from PIL import Image
from PIL import ImageTk
import numpy as np
import time
import threading
from db.db import db
from modules.nn_utility import *

class face_scan:

    def __init__(self):
        print("face scan")

def load_scan_detector_face_network(self):

    self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
    self.pb.start()

    protoPath = os.path.sep.join([self.root_path,"/models/face_detection_model", "deploy.prototxt"])
    modelPath = os.path.sep.join([self.root_path,"/models/face_detection_model",
        "res10_300x300_ssd_iter_140000.caffemodel"])

    self.net_scan_utility = nn_utility(self.cpu_type)
    self.net_scan_utility.setup_network(protoPath, modelPath,  "caffe")

    self.active_menu = 1

def run_face_scan(self,frame):
    #print("scan face")

    if self.current_id > -1:

        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detections = self.net_scan_utility.nn_detector(imageBlob)
        ts = time.time()

        self.top_label_text.set("I will take ten image of yours ;)")
        img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/takingPhotos.png"]))
        self.panel_pass.configure(image=img)
        self.panel_pass.image = img

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if self.countdown_time > 1:

                label = "Ready in...{}".format(self.countdown_time)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(frame, (10, 350 - labelSize[1]),
                                        (10 + labelSize[0], 350 + baseLine),
                                        (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

            else:

                label2 = "Shooting :)"
                labelSize, baseLine = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(frame, (10, 350 - labelSize[1]),
                                        (10 + labelSize[0], 350 + baseLine),
                                        (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label2, (10, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

            if self.total_scanned_face > 0:

                label3 = "Remaining scan : {}".format(self.total_scanned_face)
                labelSize3, baseLine3 = cv2.getTextSize(label3, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(frame, (10, 30 - labelSize3[1]),
                                        (10 + labelSize3[0], 20 + baseLine3),
                                        (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label3, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

            else:

                label3 = "All good thanks ;)"
                labelSize3, baseLine3 = cv2.getTextSize(label3, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(frame, (10, 30 - labelSize3[1]),
                                        (10 + labelSize3[0], 20 + baseLine3),
                                        (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label3, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

                self.top_label_text.set("You are Emrefied ;)")
                img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/emrefied.png"]))
                self.panel_pass.configure(image=img)
                self.panel_pass.image = img

            if confidence > 0.2:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                tstext= "%d" % ts
                if self.take_allowed == 1:
                    if self.total_scanned_face > 0:
                        humanid = "{}/dataset/humans/{}/peopled{}_.jpg".format(self.root_path,self.current_id,tstext)
                        cv2.imwrite(humanid, frame)     # save frame as JPEG file
                        print("disk write : ",humanid)
                        self.total_scanned_face -= 1
                        self.take_allowed = 0
                        self.real_countdown_time = 3
                        self.threadCount = threading.Thread(target=self.countdown, args=())
                        self.threadCount.start()

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (1, 255, 13), 2)

    return frame
