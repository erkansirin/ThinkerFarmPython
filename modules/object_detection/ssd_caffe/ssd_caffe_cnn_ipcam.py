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

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

def run_ssd_face_cnn_ipcam(self,framex):
    print("init ssd with face cnn")

    frame = imutils.resize(framex, width=525,height=480)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False, crop=False)

    self.detectorCaffe .setInput(imageBlob)
    detections = self.detectorCaffe .forward()
    if self.update_final_text == 0:
        self.T.delete("1.0", tki.END)
        self.T.insert("1.0","System ready - ThinkerFarm Face Module : detection with res10_300x300_ssd_iter_140000.caffemodel and Recognition with custom trained NN with human dataset runing on OpenCV DNN")
        self.update_final_text = 1

    cols = frame.shape[1]
    rows = frame.shape[0]

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        print("confidence : ",confidence)

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
            # Draw location of object
            color = [int(c) for c in self.COLORS[class_id]]
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          color)

            label = self.LABELSCaffe[class_id]
            print("detections : ",confidence)
            print("label : ",label)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    imageBlobFace = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detections = self.net_utility.nn_detector(imageBlobFace)
    if self.update_final_text == 0:
        self.T.delete("1.0", tki.END)
        self.T.insert("1.0","System ready - ThinkerFarm Face Module : detection with res10_300x300_ssd_iter_140000.caffemodel and Recognition with custom trained NN with human dataset runing on OpenCV DNN")
        self.update_final_text = 1

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
            vec = self.net_embedder.nn_detector(faceBlob)
            ts = time.time()


            preds = self.recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = self.le.classes_[j]
            tstext= "%d" % ts

            if proba > self.face_confidence:

                humanid = "humans/{}/peopled{}_conf_{:.2f}.jpg".format(name,tstext,proba * 100)

                text = "Staff ID : {} - {} ".format(name,person_db['people'][int(name)]['name'])
                self.top_label_text.set(text)
                img = ImageTk.PhotoImage(Image.open("ui/images/youcanpass.png")) 
                self.panel_pass.configure(image=img)
                self.panel_pass.image = img

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (1, 255, 13), 2)
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (1, 255, 13), 1)
            else:
                humanid = "data/peopled{}_conf_{:.2f}.jpg".format(tstext,proba * 100)

                text = "Staff ID : Unknown"
                self.top_label_text.set(text)
                img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/whoareyou.png"]))
                self.panel_pass.configure(image=img)
                self.panel_pass.image = img

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return frame
