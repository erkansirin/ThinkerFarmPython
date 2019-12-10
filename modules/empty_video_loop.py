# ssd_caffe.py
#
#
# Author Erkan SIRIN
# Created for ThinkerFarm project.
#
# ssd_caffe.py contains MobileNet network and runnder for the network in real-time

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

class empty_video_loop:

    def __init__(self):

        print("started")


def run_empty_video_loop(self,frame):

    frame = imutils.resize(frame, width=525,height=480)
    (h, w) = frame.shape[:2]

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
