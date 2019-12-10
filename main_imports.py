# main_imports.py
#
#
# Author Erkan SIRIN
# Created for ThinkerFarm project.
#
# main_imports.py contains requiered imports for entire application

from __future__ import print_function
from PIL import Image, ImageTk
import tkinter as tki
from tkinter import Text
from tkinter import messagebox
from tkinter import ttk
import threading
import datetime
import imutils
import cv2
import os
import pickle
import time
import numpy as np

from modules.face_recognition.face_pickle.face_pickle_ipcam import *
from modules.face_recognition.face_dlib.face_dlib import *
from modules.face_recognition.face_dlib.face_dlib_ipcam import *
from modules.object_detection.ssd_yolo.ssd_yolo import *
from modules.object_detection.ssd_caffe.ssd_caffe import *
from modules.object_detection.ssd_caffe.ssd_face_cnn import *
from modules.object_detection.ssd_caffe.ssd_caffe_cnn_ipcam import *
from modules.face_recognition.face_scan.face_scan import *
from modules.pose_estimation.edge_detection.edge_detection import *
from modules.pose_estimation.open_pose.open_pose import *
from modules.pose_estimation.open_pose.open_pose_ipcam import *
from modules.trainings.face_trainin_pickle.face_trainin_pickle import *
from modules.trainings.face_training_dlib.face_training_dlib import *
from modules.object_detection.tensorflow.faster_rcnn_tensorflow import *
from modules.classification.deepscale.squeezenet_classification import *
from modules.empty_video_loop import *
from db.person_db import person_db
from definitions.definitions import *
