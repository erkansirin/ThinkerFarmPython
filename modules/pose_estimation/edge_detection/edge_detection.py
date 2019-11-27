# edge_detection.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# edge_detection.py uses pretrained caffe model EdgeNet to detect edge in real-time

import cv2 as cv
import numpy as np
import argparse
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


class edge_detection:

    def __init__(self):

        print("posenet started")

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def init_edge_network(self):

    protoPathEdgeNet = os.path.sep.join([self.root_path,"models/edge", "deploy.prototxt"])
    modelPathEdgeNet = os.path.sep.join([self.root_path,"models/edge",
        "hed_pretrained_bsds.caffemodel"])

    self.edgeNet = cv.dnn.readNet(protoPathEdgeNet, modelPathEdgeNet)
    if self.cpu_type == 1:
        self.edgeNet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    if self.cpu_type == 2:
        self.edgeNet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


    cv.dnn_registerLayer('Crop', CropLayer)

    self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
    self.pb.start()

    text = "Show me what you got :D"

    self.top_label_text.set(text)
    img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/poseestima.png"]))
    self.panel_pass.configure(image=img)
    self.panel_pass.image = img

    self.active_menu = 8


def run_edge_network(self,frame):

    frame = imutils.resize(frame, width=525,height=480)
    (frameHeight, frameWidth) = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 30, 150)



    inp = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(350, 350),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    self.edgeNet.setInput(inp)
    print("edge net working")

    out = self.edgeNet.forward()
    #out = out[0, 0]
    #out = cv.resize(out, (frame.shape[1], frame.shape[0]))
    out = cv2.resize(out[0, 0], (frame.shape[1], frame.shape[0]))
    out = (255 * out).astype("uint8")

    return out
