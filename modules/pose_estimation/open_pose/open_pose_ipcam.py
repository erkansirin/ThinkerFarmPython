# open_pose_ipcam.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# open_pose_ipcam.py is copy of open_pose.py is real-time
# pose detection using COCO, MPI, BODY25 or HAND netowks on ipcamera

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

def run_openpose_on_ipcam(self,frame):

    frame = imutils.resize(frame, width=525,height=480)
    (frameHeight, frameWidth) = frame.shape[:2]

    inp = cv.dnn.blobFromImage(frame, self.inScale, (200, 200),
                              (0, 0, 0), swapRB=False, crop=False)
    self.detectorOpenPose.setInput(inp)
    out = self.detectorOpenPose.forward()
    print("running open poase 2")

    assert(len(self.BODY_PARTS) <= out.shape[1])

    points = []
    for i in range(len(self.BODY_PARTS)):
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        points.append((int(x), int(y)) if conf > self.pose_confidence else None)

    for pair in self.POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in self.BODY_PARTS)
        assert(partTo in self.BODY_PARTS)

        idFrom = self.BODY_PARTS[partFrom]
        idTo = self.BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = self.detectorOpenPose.getPerfProfile()
    freq = cv.getTickFrequency() / 1000

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    if self.panel is None:
        self.panel = tki.Label(self.video_frame,image=image)
        self.panel.image = image
        self.panel.pack(side="left", padx=0, pady=0)
        self.pb.stop()
        self.pb.pack_forget()

    else:
        self.panel.configure(image=image)
        self.panel.image = image
        self.panel.pack(side="left", padx=0, pady=0)

    if self.active_menu == -1:
        self.panel.pack_forget()
        self.panel = None
