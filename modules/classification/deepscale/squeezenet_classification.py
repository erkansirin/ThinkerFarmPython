# squeezenet_classification.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# squeezenet_classification.py OpenCV dnn implementation of DeepScale SqueezeNet in realtime

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

class squeezenet_classification:

    def __init__(self):

        print("started")

def load_squeezenet_classification_network(self):

    self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
    self.pb.start()

    print("AI Edge : Loading DeepScale SqueezeNet Network")
    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Loading DeepScale SqueezeNet Network")

    squeezenet_config_path = os.path.sep.join([self.root_path,"/models/classification/squeezenet", "squeezenet_v1.1.prototxt"])
    squeezenet_path = os.path.sep.join([self.root_path,"/models/classification/squeezenet",
        "squeezenet_v1.1.caffemodel"])

    squeezenet_labels_path  = os.path.sep.join([self.root_path,"/models/classification/squeezenet", "squeezenet.names"])
    self.squeezenet_labels = open(squeezenet_labels_path ).read().strip().split("\n")
    self.detector_squeezenet = cv2.dnn.readNetFromCaffe(squeezenet_config_path , squeezenet_path)
    if self.cpu_type == 1:
        self.detector_squeezenet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    if self.cpu_type == 2:
        self.detector_squeezenet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    np.random.seed(100)
    self.COLORS = np.random.randint(0, 255, size=(100, 3),
    dtype="uint8")

    self.active_menu = 12

def run_squeezenet_classification(self,frame):

    frame = imutils.resize(frame, width=525,height=480)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(525,480)), 1, (525, 480), (0, 0, 0), swapRB=False, crop=False)
    #imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False, crop=False)

    self.detector_squeezenet .setInput(imageBlob)
    detections = self.detector_squeezenet .forward()

    cols = frame.shape[1]
    rows = frame.shape[0]

    for i in range(len(detections[0])):

        confidences = detections[0]
        confidence = confidences[i]

        if confidence > self.classifier_confidence:

            class_id = int(confidences[i])

            label = self.squeezenet_labels[i]
            print("class name : ",label)
            print("squeezenet detections confidence: ",confidence)

            labelText = "Detected : {} with confidence :{}".format(label,confidence)


            self.T.delete("1.0", tki.END)
            self.T.insert("1.0",labelText)


    return frame
