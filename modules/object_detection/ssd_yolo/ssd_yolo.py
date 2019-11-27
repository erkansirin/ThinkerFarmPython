# ssd_yolo.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# ssd_yolo.py contains loader and runnder for Tiny Yolo object detection network

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

class ssd_yolo:

    def __init__(self):
        print("ssd_yolo")

def load_ssd_yolo_network(self):

    labelsPath = os.path.sep.join([self.root_path,"/models/object_detection/yolo-coco", "coco.names"])
    self.LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
    dtype="uint8")
    weightsPath = os.path.sep.join([self.root_path,"models/object_detection/yolo-coco", "yolov3-spp.weights"])
    configPath = os.path.sep.join([self.root_path,"models/object_detection/yolo-coco", "yolov3-spp.cfg"])

    print("AI Edge : Loading Yolo Network")
    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Loading Yolo Network")

    self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    if self.cpu_type == 0:
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    else:
        self.net.setPreferableTarget(cv2.dnn.DNN_BACKEND_HALIDE)
    self.ln = self.net.getLayerNames()
    self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    self.active_menu = 3

def run_ssd_yolo(self,frame):

    frame = imutils.resize(frame, width=525,height=480)
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (416, 416)), 1 / 255.0, (416, 416), swapRB=True, crop=False)

    self.net.setInput(blob)
    start = time.time()
    layerOutputs = self.net.forward(self.ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.3:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                0.3)

                if len(idxs) > 0:

                    for i in idxs.flatten():

                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                        confidences[i])
                        cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame
