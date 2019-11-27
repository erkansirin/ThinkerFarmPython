# open_pose_with_ssd.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# open_pose_with_ssd is copy of open_pose.py is real-time
# pose detection using COCO, MPI, BODY25 or HAND netowks
# and it uses human detection to apply network on multple layers for more than one person


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

def setupOpenPoseNetWithSSD(self):

    if self.pose_dataset == 'COCO':
        self.BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                       "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

        self.POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                       ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                       ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                       ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                       ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    if self.pose_dataset == 'MPI':
        self.BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                       "Background": 15 }

        self.POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                       ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                       ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                       ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    if self.pose_dataset == 'BODY25':
        self.BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                       "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                       "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                       "Background": 15 }

        self.POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                       ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                       ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                       ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]


    if self.pose_dataset == 'HAND':
        self.BODY_PARTS = { "Wrist": 0,
                       "ThumbMetacarpal": 1, "ThumbProximal": 2, "ThumbMiddle": 3, "ThumbDistal": 4,
                       "IndexFingerMetacarpal": 5, "IndexFingerProximal": 6, "IndexFingerMiddle": 7, "IndexFingerDistal": 8,
                       "MiddleFingerMetacarpal": 9, "MiddleFingerProximal": 10, "MiddleFingerMiddle": 11, "MiddleFingerDistal": 12,
                       "RingFingerMetacarpal": 13, "RingFingerProximal": 14, "RingFingerMiddle": 15, "RingFingerDistal": 16,
                       "LittleFingerMetacarpal": 17, "LittleFingerProximal": 18, "LittleFingerMiddle": 19, "LittleFingerDistal": 20,
                     }

        self.POSE_PAIRS = [ ["Wrist", "ThumbMetacarpal"], ["ThumbMetacarpal", "ThumbProximal"],
                       ["ThumbProximal", "ThumbMiddle"], ["ThumbMiddle", "ThumbDistal"],
                       ["Wrist", "IndexFingerMetacarpal"], ["IndexFingerMetacarpal", "IndexFingerProximal"],
                       ["IndexFingerProximal", "IndexFingerMiddle"], ["IndexFingerMiddle", "IndexFingerDistal"],
                       ["Wrist", "MiddleFingerMetacarpal"], ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
                       ["MiddleFingerProximal", "MiddleFingerMiddle"], ["MiddleFingerMiddle", "MiddleFingerDistal"],
                       ["Wrist", "RingFingerMetacarpal"], ["RingFingerMetacarpal", "RingFingerProximal"],
                       ["RingFingerProximal", "RingFingerMiddle"], ["RingFingerMiddle", "RingFingerDistal"],
                       ["Wrist", "LittleFingerMetacarpal"], ["LittleFingerMetacarpal", "LittleFingerProximal"],
                       ["LittleFingerProximal", "LittleFingerMiddle"], ["LittleFingerMiddle", "LittleFingerDistal"] ]

    self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
    self.pb.start()

    self.inWidth = 525
    self.inHeight = 480
    self.inScale = 0.003922

    print("AI Edge : Loading OpenPose Network 2")
    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","AI Edge : Loading OpenPose Network 2")

    if self.pose_dataset == 'COCO':
        protoPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/pose/coco", "pose_deploy_linevec.prototxt"])
        modelPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/pose/coco",
            "pose_iter_440000.caffemodel"])


    if self.pose_dataset == 'MPI':
        protoPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/pose/mpi", "pose_deploy_linevec_faster_4_stages.prototxt"])
        modelPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/pose/mpi",
            "pose_iter_160000.caffemodel"])

    if self.pose_dataset == 'BODY25':
        protoPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/pose/body_25", "pose_deploy.prototxt"])
        modelPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/pose/body_25",
            "pose_iter_584000.caffemodel"])

    if self.pose_dataset == 'HAND':
        protoPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/hand", "pose_deploy.prototxt"])
        modelPathOpenPose = os.path.sep.join([self.root_path,"/models/open_pose_net/hand",
            "pose_iter_102000.caffemodel"])


    self.detectorOpenPose = cv2.dnn.readNet(protoPathOpenPose , modelPathOpenPose )
    if self.cpu_type == 1:
        self.detectorOpenPose.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    if self.cpu_type == 2:
        self.detectorOpenPose.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


    protoPathCaffe = os.path.sep.join([self.root_path,"/models/ssd_models/caffe", "MobileNetSSD_deploy.prototxt"])
    modelPathCaffe = os.path.sep.join([self.root_path,"/models/ssd_models/caffe",
        "MobileNetSSD_deploy.caffemodel"])



    labelsPathCaffe  = os.path.sep.join([self.root_path,"/models/ssd_models/caffe", "caffe.names"])
    self.LABELSCaffe = open(labelsPathCaffe ).read().strip().split("\n")
    self.detectorCaffe = cv2.dnn.readNetFromCaffe(protoPathCaffe , modelPathCaffe )
    if self.cpu_type == 1:
        self.detectorCaffe.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    if self.cpu_type == 2:
        self.detectorCaffe.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


    text = "Show me what you got :D"


    self.top_label_text.set(text)
    img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "images/poseestima.png"]))
    self.panel_pass.configure(image=img)
    self.panel_pass.image = img

    self.active_menu = 7

def runOpenposeWithSSD(self,frame):

    frame = imutils.resize(frame, width=525,height=480)
    (frameHeight, frameWidth) = frame.shape[:2]
    print("running open poase")

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False, crop=False)

    cv2.dnn.resetMyriadDevice()
    self.detectorCaffe.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

    self.detectorCaffe .setInput(imageBlob)
    detections = self.detectorCaffe .forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        print("confidence : ",confidence)

        if confidence > self.ssdCaffeConfidence:
            class_id = int(detections[0, 0, i, 1])

            if class_id == 15:

                box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]


                inp = cv.dnn.blobFromImage(face, self.inScale, (200, 200),
                                          (0, 0, 0), swapRB=False, crop=False)

                cv2.dnn.resetMyriadDevice()
                self.detectorOpenPose.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)


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
        #print("self panel create")
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
