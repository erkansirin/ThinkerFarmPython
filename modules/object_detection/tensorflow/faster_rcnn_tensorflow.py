# faster_rcnn_tensorflow.py
#
#
# Author Erkan SIRIN
# Created for ThinkerFarm project.
#
# faster_rcnn_tensorflow.py contains dnn
# implementation of tensorflow object detection network

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

class faster_rcnn_tensorflow:

    def __init__(self):

        print("started")

def load_faster_rcnn_tensorflow_network(self):

    self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
    self.pb.start()

    print("ThinkerFarm : Loading Faster RCNNensorflow Network")
    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","ThinkerFarm : Loading Faster RCNN TensorFlow Network")

    pbtxtPath = "models/object_detection/tensorflow/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    pbPath = "models/object_detection/tensorflow/faster_rcnn_inception_v2_coco_2018_01_28.pb"

    labelsPathCaffe  = "models/object_detection/tensorflow/faster_rcnn_inception_v2_coco_2018_01_28.names"
    self.LABELSCaffe = open(labelsPathCaffe ).read().strip().split("\n")
    self.detector_frcnn_tensorflow = cv2.dnn.readNet(pbtxtPath , pbPath, 'tensorflow')
    if self.cpu_type == 1:
        self.detector_frcnn_tensorflow.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    if self.cpu_type == 2:
        self.detector_frcnn_tensorflow.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    np.random.seed(100)
    self.COLORS = np.random.randint(0, 255, size=(100, 3),
    dtype="uint8")

    self.active_menu = 11

def run_faster_rcnn_tensorflow(self,frame):



    frame = imutils.resize(frame, width=525,height=480)
    (h, w) = frame.shape[:2]
    #frame = frame[:, :, ::-1]



    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300, 300), (0, 0, 0), swapRB=False, crop=False)
    #imageBlob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)

    #imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False, crop=False)


    self.detector_frcnn_tensorflow .setInput(imageBlob)
    print("setInput : ")
    detections = self.detector_frcnn_tensorflow .forward()
    print("forward : ")
    if self.update_final_text == 0:
        self.T.delete("1.0", tki.END)
        self.T.insert("1.0","System ready - ThinkerFarm Face Module : detection with res10_300x300_ssd_iter_140000.caffemodel and Recognition with custom trained NN with human dataset runing on OpenCV DNN")
        self.update_final_text = 1

    cols = frame.shape[1]
    rows = frame.shape[0]

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > self.ssd_caffe_confidence:

            class_id = int(detections[0, 0, i, 1])

            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)

            heightFactor = frame.shape[0]/410.0
            widthFactor = frame.shape[1]/525.0

            xLeftBottom = int(widthFactor * xLeftBottom)
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)

            color = [int(c) for c in self.COLORS[class_id]]
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          color)

            print("class_id : ",class_id)

            label = self.LABELSCaffe[class_id]

            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))



    return frame
