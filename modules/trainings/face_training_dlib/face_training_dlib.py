# import the necessary packages
from imutils import paths
from modules.face_recognition.face_dlib.face_dlib_utility import *
import argparse
import pickle
import cv2
import os
import time
import tkinter as tki
import numpy as np
import imutils
import requests
import threading
import datetime
from datetime import date, timedelta
from pathlib import Path
import gc



def face_training_dlib(self):


	dlib.DLIB_USE_CUDA = True

	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images("dataset/humans"))

	knownEncodings = []
	knownNames = []


	for (i, imagePath) in enumerate(imagePaths):

		labelText = "ThinkerFarm : processing image Dlib {}/{} - Path : {}".format(i + 1,
			len(imagePaths),imagePath)
		print("labelText : ",labelText)

		name = imagePath.split(os.path.sep)[-2]


		image = cv2.imread(imagePath)

		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

		boxes = face_locations(rgb, model="cnn")

		encodings = face_encodings(rgb, boxes)

		for encoding in encodings:

			knownEncodings.append(encoding)
			knownNames.append(name)


		del labelText
		del name
		del image
		del rgb
		del boxes
		del encodings
		gc.collect()
		time.sleep(5.0)


	print("ThinkerFarm: serializing encodings...")
	data = {"encodings": knownEncodings, "names": knownNames}
	f = open("models/face_recognition_models/encodings.pickle", "wb")
	f.write(pickle.dumps(data))
	f.close()

	print("ThinkerFarm: finished training dlib...")

	load_dlib_face_network_shell(self)

def load_dlib_face_network_shell(self):

    self.T.delete("1.0", tki.END)
    self.T.insert("1.0","ThinkerFarm : Loading Dlib Face Network")

    print("ThinkerFarm : Loading Dlib Face Network")

    dlib.DLIB_USE_CUDA = True

    config = Path(dlib_encodings)

    if config.is_file():
        self.dataEncodings = pickle.loads(open(os.path.sep.join([dlib_encodings]), "rb").read())
        self.active_menu = 2
    else:
        print("pickle does not exist")

    print("ThinkerFarm : Running Face Dlib")

    self.thread_count_training = threading.Thread(target=self.countdown_train, args=())
    self.thread_count_training.start()

def download_dataset(self):

	face_training_dlib(self)
