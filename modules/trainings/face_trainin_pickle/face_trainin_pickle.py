# face_training_pickle.py
#
#
# Author by Erkan SIRIN
# Created for AI Edge project.
#
# face_training_pickle.py extracts face embeddings from
# humans dataset and train network with embbedings

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import tkinter as tki
# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
import os
from modules.face_recognition.face_pickle.face_pickle import *


def extract_pickle_embeddings(self):

	self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
	self.pb.start()

	print("AI Edge :  Init Face Detetctor")
	print("AI Edge : quantifying faces")
	imagePaths = list(paths.list_images(os.path.sep.join([self.root_path,"dataset/humans"])))

	knownEmbeddings = []
	knownNames = []

	total = 0

	self.top_label_text.set("Learning New Faces... =)")
	img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/workingbusy.png"]))
	self.panel_pass.configure(image=img)
	self.panel_pass.image = img


	for (i, imagePath) in enumerate(imagePaths):
		name = imagePath.split(os.path.sep)[-2]

		image = cv2.imread(imagePath)
		labelText = "AI Edge : processing image {}/{} - Path : {}".format(i + 1,
			len(imagePaths),imagePath)

		self.T.delete("1.0", tki.END)
		self.T.insert("1.0",labelText)

		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		self.detector.setInput(imageBlob)
		detections = self.detector.forward()

		if len(detections) > 0:

			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			if confidence > 0.2:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				if fW < 20 or fH < 20:
					continue

				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				self.embedder.setInput(faceBlob)
				vec = self.embedder.forward()

				print("found name :",name)
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

	print("AI Edge : serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open(os.path.sep.join([self.root_path,"/models/face_recognition_models",
        "embeddings.pickle"]), "wb")
	f.write(pickle.dumps(data))
	f.close()
	train_face_pickle(self)


def train_face_pickle(self):

	print("AI Edge :  Loading Face Details")
	data = pickle.loads(open(os.path.sep.join([self.root_path,"/models/face_recognition_models",
        "embeddings.pickle"]), "rb").read())

	print("AI Edge : encoding labels...")
	self.T.delete("1.0", tki.END)
	self.T.insert("1.0","AI Edge : encoding labels..")

	le = LabelEncoder()
	labels = le.fit_transform(data["names"])
	print("data[names] :",data["names"])
	print("labels : ",labels)

	self.T.delete("1.0", tki.END)
	self.T.insert("1.0","AI Edge : training model..")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

	f = open(os.path.sep.join([self.root_path,"/models/face_recognition_models",
        "recognizer.pickle"]), "wb")
	f.write(pickle.dumps(recognizer))
	f.close()

	f = open(os.path.sep.join([self.root_path,"/models/face_recognition_models",
        "le.pickle"]), "wb")
	f.write(pickle.dumps(le))
	f.close()

	self.top_label_text.set("Well done training Emrefied... :D =)")
	img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/emrefied.png"]))
	self.panel_pass.configure(image=img)
	self.panel_pass.image = img

	load_pickle_face_network(self)
