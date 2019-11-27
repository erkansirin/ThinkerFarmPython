# import the necessary packages
from imutils import paths
from modules.face_recognition.face_dlib.face_dlib_utility import *
import argparse
import pickle
import cv2
import os

import tkinter as tki
import numpy as np
import imutils
import requests
import threading
import datetime
from datetime import date, timedelta
from modules.face_recognition.face_dlib.face_dlib_shell import *
from pathlib import Path
import gc



def face_training_dlib(self):



	# self.pb.pack(expand=True, fill=tki.BOTH, padx=[200,0])
	# self.pb.start()
	#
	# self.top_label_text.set("Learning New Faces... =)")
	# img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/workingbusy.png"]))
	# self.panel_pass.configure(image=img)
	# self.panel_pass.image = img

	# construct the argument parser and parse the arguments


	# grab the paths to the input images in our dataset

	dlib.DLIB_USE_CUDA = True

	print("[INFO] quantifying faces...")
	imagePaths = list(paths.list_images(os.path.sep.join([self.root_path,"dataset/humans"])))


	# initialize the list of known encodings and known names
	knownEncodings = []
	knownNames = []


	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path


		labelText = "AI Edge : processing image Dlib {}/{} - Path : {}".format(i + 1,
			len(imagePaths),imagePath)
		print("labelText : ",labelText)

		# self.T.delete("1.0", tki.END)
		# self.T.insert("1.0",labelText)


		name = imagePath.split(os.path.sep)[-2]
		print("debug image error : 1")



		# load the input image and convert it from BGR (OpenCV ordering)
		# to dlib ordering (RGB)
		image = cv2.imread(imagePath)
		#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		print("debug image error : 2")
		boxes = face_locations(rgb, model="cnn")

		encodings = face_encodings(rgb, boxes)
		print("debug image error : 3")

		# loop over the encodings
		for encoding in encodings:
			# add each encoding + name to our set of known names and
			# encodings
			print("debug image error : 1")
			knownEncodings.append(encoding)
			knownNames.append(name)


	    	# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
		#try:
			#boxes = face_locations(rgb, model="hog")
			# print("boxes.count : ",len(boxes))
			# if len(boxes) > 0:
			# 	top = boxes[0][0]
			# 	right = boxes[0][1]
			# 	bottom = boxes[0][2]
			# 	left = boxes[0][3]
			# 	ts = time.time()
			# 	tstext= "%d"%ts
			# 	crop_img = image[top:bottom,left:right]
			# 	humanid = "{}/dataset/trained/crop_{}_.jpg".format(self.root_path,tstext)
			# 	cv2.imwrite(humanid, crop_img)

			# for (top, right, bottom, left) in zip(boxes[0]):
			# 	ts = time.time()
			# 	tstext= "%d"%ts
			# 	crop_img = frame[top:bottom,left:right]
			# 	humanid = "{}/dataset/trained/crop_{}_.jpg".format(self.root_path,tstext)
			# 	cv2.imwrite(humanid, crop_img)


			# compute the facial embedding for the face
		# 	if len(boxes) > 0:
		# 		encodings = face_encodings(rgb, boxes)
		# 		print("debug image error : 3")
		#
		# 		# loop over the encodings
		# 		for encoding in encodings:
		# 			# add each encoding + name to our set of known names and
		# 			# encodings
		# 			print("debug image error : 1")
		# 			knownEncodings.append(encoding)
		# 			knownNames.append(name)
		# except RuntimeError as e:
		# 	print("RuntimeError")

		del labelText
		del name
		del image
		del rgb
		del boxes
		del encodings
		gc.collect()
		time.sleep(5.0)




	# dump the facial encodings + names to disk
	print("[INFO] serializing encodings...")
	data = {"encodings": knownEncodings, "names": knownNames}
	f = open(os.path.sep.join([self.root_path,"/models/face_recognition_models",
        "encodings.pickle"]), "wb")
	f.write(pickle.dumps(data))
	f.close()

	# self.top_label_text.set("Well done training Emrefied... :D =)")
	# img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/emrefied.png"]))
	# self.panel_pass.configure(image=img)
	# self.panel_pass.image = img

	print("finished training dlib...")

	# api-endpoint
	URL = "http://139.162.142.162:1994/api/companies/1/training-mode?trainingMode=false"


	# defining a params dict for the parameters to be sent to the API
	PARAMS = {}

	# sending get request and saving the response as response object
	r = requests.post(url = URL, params = PARAMS)

	# extracting data in json format
	#data = r.json()
	#print("response :",data)
	# self.active_menu = 2
	# self.thread_count_training = threading.Thread(target=self.countdown_train, args=())
	# self.thread_count_training.start()

	load_dlib_face_network_shell(self)

def download_dataset(self):

	face_training_dlib(self)


	# api-endpoint

	# yesterday = date.today() - timedelta(days=1)
	#
	#
	# now = datetime.datetime.now()
	# date_time = yesterday.strftime("%d-%m-%Y %H:%M:%S")
	# #date_time = "01-01-2018 01:01:01"
	# print("date_time : ",date_time)
	#
	# URL = "http://139.162.142.162:1994/api/companies/1/employees?updated=%s"%date_time
	# print("date_time URL: ",URL)
	#
	# # location given here
	# location = "employee"
	#
	# # defining a params dict for the parameters to be sent to the API
	# PARAMS = {'id':0, 'name':'String'}
	#
	# # sending get request and saving the response as response object
	# r = requests.get(url = URL, params = PARAMS)
	#
	# # extracting data in json format
	# data = r.json()
	# print("response :",data)
	#
	# config = Path('/home/erkan/Desktop/ai/edge/models/face_recognition_models/encodings.pickle')
	#
	# if config.is_file():
	# 	remove_model = 'rm /home/erkan/Desktop/ai/edge/models/face_recognition_models/encodings.pickle'
	# 	os.system(remove_model)
	# else:
	# 	print("pickle does not exist")
	#
	#
	#
	#
	# remove_dataset = 'rm -r /home/erkan/Desktop/ai/edge/dataset/humans'
	# os.system(remove_dataset)
	# create_dataset = 'mkdir /home/erkan/Desktop/ai/edge/dataset/humans'
	# os.system(create_dataset)
	# #enter_dataset = 'cd /home/erkan/Desktop/ai/ai/edge/dataset/humans'
	# #os.system(enter_dataset)
	# total_data_count = len(data)
	# downloaded_file_count = 0
	# for personids in data:
	# 	print("data[x] : ",personids["id"])
	# 	create_person_id = 'mkdir /home/erkan/Desktop/ai/edge/dataset/humans/%s'%personids["id"]
	# 	os.system(create_person_id)
	# 	downloaded_file_count +=1
	# 	for person_images in personids["images"]:
	# 		print("person_images : ",person_images)
	#
	# 		img_data = requests.get(person_images["url"]).content
	# 		with open('/home/erkan/Desktop/ai/edge/dataset/humans/%s/%s.jpg'%(personids["id"],person_images["id"]), 'wb') as handler:
	# 		    handler.write(img_data)
	# 		time.sleep(5.0)
	# 	time.sleep(5.0)
	#
	#
	# if downloaded_file_count == total_data_count:
	# 	face_training_dlib(self)
