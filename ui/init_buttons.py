# init_buttons.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# init_buttons.py contains button objects for mainloop

from db.db import db
import os
from tkinter import Text
from tkinter import messagebox
from tkinter import ttk


def init_buttons(self, tki):


    self.main_button_frame = tki.Frame(self.root)
    self.main_button_frame.pack(fill=tki.Y, side=tki.RIGHT)
    self.main_button_frame.columnconfigure(0, weight=1)
    self.main_button_frame.columnconfigure(1, weight=1)
    #self.main_button_frame.configure(bg='white')

    self.button_frame = tki.Frame(self.main_button_frame)
    self.button_frame.pack(fill=tki.Y, side=tki.RIGHT)
    self.button_frame.columnconfigure(0, weight=1)
    self.button_frame.columnconfigure(1, weight=1)
    #self.button_frame.configure(bg='white')



    self.scan_faces_button = tki.Button(self.button_frame, text="Scan My Face", height = 2, width = 20, command = self.open_face_scan_menu, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.scan_faces_button.pack(side=tki.TOP, padx=[0,0])

    self.face_implementation_button = tki.Button(self.button_frame, text="Face Implementations", height = 2, width =20, command = self.open_face_implementation_menu, font = ('Comic Sans MS',15),
    bg = 'SkyBlue3', fg = 'white', anchor='center')
    self.face_implementation_button.pack(side=tki.TOP, padx=[0,0])

    self.ssd_menu_button = tki.Button(self.button_frame, text="Object Detection SSD", height = 2, width =20, command = self.open_ssd_main, font = ('Comic Sans MS',15),
    bg = 'SkyBlue3', fg = 'white', anchor='center')
    self.ssd_menu_button.pack(side=tki.TOP, padx=[0,0])

    self.image_classifiers_button = tki.Button(self.button_frame, text="Classifiers", height = 2, width =20, command = self.open_image_classifier_main, font = ('Comic Sans MS',15),
    bg = 'SkyBlue3', fg = 'white', anchor='center')
    self.image_classifiers_button.pack(side=tki.TOP, padx=[0,0])

    self.pose_estimation_button = tki.Button(self.button_frame, text="Pose Estimation", height = 2, width =20, command = self.open_pose_estimation, font = ('Comic Sans MS',15),
    bg = 'SkyBlue3', fg = 'white', anchor='center')
    self.pose_estimation_button.pack(side=tki.TOP, padx=[0,0])

    self.gans_button = tki.Button(self.button_frame, text="GANs", height = 2, width =20, command = self.open_gans_main, font = ('Comic Sans MS',15),
    bg = 'SkyBlue3', fg = 'white', anchor='center')
    self.gans_button.pack(side=tki.TOP, padx=[0,0])

    self.training_button = tki.Button(self.button_frame, text="Training", height = 2, width =20, command = self.open_training_menu, font = ('Comic Sans MS',15),
    bg = 'SkyBlue3', fg = 'white', anchor='center')
    self.training_button.pack(side=tki.TOP, padx=[0,0])

    self.exit_button = tki.Button(self.button_frame, text="Exit", height = 2, width =20, command = self.close_app, font = ('Comic Sans MS',15),
    bg = 'SkyBlue3', fg = 'white', anchor='center')
    self.exit_button.pack(side=tki.TOP, padx=[0,0])


    self.classifiers_button_frame = tki.Frame(self.main_button_frame)
    self.classifiers_button_frame.columnconfigure(0, weight=1)
    self.classifiers_button_frame.columnconfigure(1, weight=1)
    self.classifiers_button_frame.configure(bg='white')

    self.classifiers_back_button = tki.Button(self.classifiers_button_frame, text="Back", height = 2, width = 20, command = self.back_from_classifiers, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.classifiers_back_button.pack(side=tki.TOP, padx=[0,0])

    self.squeezenet_button = tki.Button(self.classifiers_button_frame, text="DeepScale SqueezeNet ", height = 2, width = 20, command = self.start_squeezenet, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.squeezenet_button.pack(side=tki.TOP, padx=[0,0])


    self.ssd_button_frame = tki.Frame(self.main_button_frame)
    self.ssd_button_frame.columnconfigure(0, weight=1)
    self.ssd_button_frame.columnconfigure(1, weight=1)
    self.ssd_button_frame.configure(bg='white')

    self.ssd_back_button = tki.Button(self.ssd_button_frame, text="Back", height = 2, width = 20, command = self.back_from_ssd, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.ssd_back_button.pack(side=tki.TOP, padx=[0,0])

    self.yolo_button = tki.Button(self.ssd_button_frame, text="SSD Yolo", height = 2, width = 20, command = self.start_ssd_yolo, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.yolo_button.pack(side=tki.TOP, padx=[0,0])

    self.mobilenet_caffe_button = tki.Button(self.ssd_button_frame, text="MobileNet Caffe", height = 2, width = 20, command = self.start_mobilenet_caffe, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.mobilenet_caffe_button.pack(side=tki.TOP, padx=[0,0])

    self.ssd_and_face_button = tki.Button(self.ssd_button_frame, text="SSD+Face+CNN", height = 2, width = 20, command = self.start_ssd_face_cnn, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.ssd_and_face_button.pack(side=tki.TOP, padx=[0,0])

    self.ssd_and_face_ipcam_button = tki.Button(self.ssd_button_frame, text="IPCam SSD+Face CNN", height = 2, width = 20, command = self.start_ssd_face_cnn_ipcam, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.ssd_and_face_ipcam_button.pack(side=tki.TOP, padx=[0,0])

    self.faster_rcnn_tensorflow_button = tki.Button(self.ssd_button_frame, text="Faster RCNN_Tensorflow", height = 2, width = 20, command = self.start_faster_rcnn_tensorflow, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.faster_rcnn_tensorflow_button.pack(side=tki.TOP, padx=[0,0])


    self.training_button_frame = tki.Frame(self.main_button_frame)
    self.training_button_frame.columnconfigure(0, weight=1)
    self.training_button_frame.columnconfigure(1, weight=1)
    self.training_button_frame.configure(bg='white')

    self.training_back_button = tki.Button(self.training_button_frame, text="Back", height = 2, width = 20, command = self.back_from_training, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.training_back_button.pack(side=tki.TOP, padx=[0,0])

    self.train_face_button = tki.Button(self.training_button_frame, text="Train new faces", height = 2, width = 20, command = self.start_face_training, font = ('Comic Sans MS',15),
    bg = 'orchid3', fg = 'white', anchor='center')
    self.train_face_button.pack(side=tki.TOP, padx=[0,0])

    self.train_face_dlib_button = tki.Button(self.training_button_frame, text="Train new faces dlib", height = 2, width = 20, command = self.start_face_training_dlib, font = ('Comic Sans MS',15),
    bg = 'orchid3', fg = 'white', anchor='center')
    self.train_face_dlib_button.pack(side=tki.TOP, padx=[0,0])



    self.posenet_button_frame = tki.Frame(self.main_button_frame)
    self.posenet_button_frame.columnconfigure(0, weight=1)
    self.posenet_button_frame.columnconfigure(1, weight=1)
    self.posenet_button_frame.configure(bg='white')

    self.posenet_back_button = tki.Button(self.posenet_button_frame, text="Back", height = 2, width = 20, command = self.back_from_pose_estimation, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.posenet_back_button.pack(side=tki.TOP, padx=[0,0])

    self.posenet_coco_button = tki.Button(self.posenet_button_frame, text="OpenPose COCO", height = 2, width = 20, command = self.start_posenet_coco, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.posenet_coco_button.pack(side=tki.TOP, padx=[0,0])

    self.posenet_mpi_button = tki.Button(self.posenet_button_frame, text="OpenPose MPI", height = 2, width = 20, command = self.start_posenet_mpi, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.posenet_mpi_button.pack(side=tki.TOP, padx=[0,0])

    self.posenet_mpi_ipcam_button = tki.Button(self.posenet_button_frame, text="OpenPose MPI IPCam", height = 2, width = 20, command = self.start_posenet_mpi_ipcam, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.posenet_mpi_ipcam_button.pack(side=tki.TOP, padx=[0,0])

    self.posenet_body_button = tki.Button(self.posenet_button_frame, text="OpenPose BODY25 DNN", height = 2, width = 20, command = self.start_posenet_body, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.posenet_body_button.pack(side=tki.TOP, padx=[0,0])

    self.posenet_hand_button = tki.Button(self.posenet_button_frame, text="OpenPose HAND", height = 2, width = 20, command = self.start_posenet_hand, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.posenet_hand_button.pack(side=tki.TOP, padx=[0,0])

    self.edge_detection_button = tki.Button(self.posenet_button_frame, text="Edge Detection", height = 2, width = 20, command = self.start_edge_detection, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.edge_detection_button.pack(side=tki.TOP, padx=[0,0])

    self.posenet_mpi_ssd_button = tki.Button(self.posenet_button_frame, text="OpenPose MPI+SSD", height = 2, width = 20, command = self.start_posenet_mpi_with_ssd, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.posenet_mpi_ssd_button.pack(side=tki.TOP, padx=[0,0])

    self.poseTensorflowButton = tki.Button(self.posenet_button_frame, text="PoseNet Tensorflow", height = 2, width = 20, command = self.callPoseTensor, font = ('Comic Sans MS',15),
    bg = 'DarkOrange1', fg = 'white', anchor='center')
    self.poseTensorflowButton.pack(side=tki.TOP, padx=[0,0])

    self.face_implementation_button_frame = tki.Frame(self.main_button_frame)
    self.face_implementation_button_frame.columnconfigure(0, weight=1)
    self.face_implementation_button_frame.columnconfigure(1, weight=1)
    #self.face_implementation_button_frame.configure(bg='white')

    self.face_implementation_back_button = tki.Button(self.face_implementation_button_frame, text="Back", height = 2, width = 20, command = self.back_from_face_implementations, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.face_implementation_back_button.pack(side=tki.TOP, padx=[0,0])

    self.face_pickle_button = tki.Button(self.face_implementation_button_frame, text="Face Recognition Pickle", height = 2, width = 20, command = self.start_face_pickle, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.face_pickle_button.pack(side=tki.TOP, padx=[0,0])

    self.face_dlib_button = tki.Button(self.face_implementation_button_frame, text="Face Recognition Dlib", height = 2, width = 20, command = self.start_face_dlib, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.face_dlib_button.pack(side=tki.TOP, padx=[0,0])

    self.face_dlib_dnn_button = tki.Button(self.face_implementation_button_frame, text="Face Dlib+CNN", height = 2, width = 20, command = self.start_face_dlib_ssd, font = ('Comic Sans MS',15),
    bg = 'red2', fg = 'white', anchor='center')
    self.face_dlib_dnn_button.pack(side=tki.TOP, padx=[0,0])

    self.face_dlib_ipcam_button = tki.Button(self.face_implementation_button_frame, text="Face Dlib+IPCam", height = 2, width = 20, command = self.start_face_dlib_ipcam, font = ('Comic Sans MS',15),
    bg = 'red2', fg = 'white', anchor='center')
    self.face_dlib_ipcam_button.pack(side=tki.TOP, padx=[0,0])

    self.face_pickle_ipcam_button = tki.Button(self.face_implementation_button_frame, text="Face Pickle+IPCam", height = 2, width = 20, command = self.start_face_pickle_ipcam, font = ('Comic Sans MS',15),
    bg = 'red2', fg = 'white', anchor='center')
    self.face_pickle_ipcam_button.pack(side=tki.TOP, padx=[0,0])


    self.face_pickle_twocam_button = tki.Button(self.face_implementation_button_frame, text="Face Two Cam NoView", height = 2, width = 20, command = self.start_face_twocam_noview, font = ('Comic Sans MS',15),
    bg = 'red2', fg = 'white', anchor='center')
    self.face_pickle_twocam_button.pack(side=tki.TOP, padx=[0,0])

    self.scan_menu_button_frame = tki.Frame(self.main_button_frame)
    self.scan_menu_button_frame.columnconfigure(0, weight=1)
    self.scan_menu_button_frame.columnconfigure(1, weight=1)
    #self.scan_menu_button_frame.configure(bg='white')

    self.scan_back_button = tki.Button(self.scan_menu_button_frame, text="Back", height = 2, width = 20, command = self.back_from_scan_menu, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.scan_back_button.pack(side=tki.TOP, padx=[0,0])

    self.list_box_frame = tki.Frame(self.scan_menu_button_frame)
    self.list_box_frame.pack(fill=tki.Y, side=tki.TOP)
    self.list_box_frame.columnconfigure(0, weight=1)
    self.list_box_frame.columnconfigure(1, weight=1)
    self.list_box_frame.configure(bg='white')

    self.menu_scroll_bar = tki.Scrollbar(self.list_box_frame)
    self.list = tki.Listbox(self.list_box_frame, height = 13, font = ('Comic Sans MS',15), yscrollcommand = self.menu_scroll_bar.set )
    for line in range(len(db['people'])):
        self.list.insert(tki.END, db['people'][int(line)]['name'])
        self.list.itemconfig("end", bg = "chartreuse4")

    self.list.bind('<<ListboxSelect>>', self.list_item_selected)

    self.menu_scroll_bar.pack( side = tki.RIGHT, fill = tki.Y )

    self.list.pack( side = tki.TOP, fill = tki.Y )

    self.menu_scroll_bar.config( command = self.list.yview )

    self.scan_back_button = tki.Button(self.scan_menu_button_frame, text="Start Scan", height = 2, width = 20, command = self.start_scan, font = ('Comic Sans MS',15),
    bg = 'chartreuse3', fg = 'white', anchor='center')
    self.scan_back_button.pack(side=tki.TOP, padx=[0,0])

    self.bottom_status_frame = tki.Frame(self.root)
    self.bottom_status_frame.pack(side=tki.BOTTOM, fill=tki.X)
    self.bottom_status_frame.configure(bg='white')

    self.T = tki.Text(self.bottom_status_frame, height=3, width=30)
    self.T.pack(side=tki.RIGHT, fill=tki.X, expand="yes", padx=0, pady=0)
    self.T.insert("1.0","System ready - AI Edge Face Module : detection with res10_300x300_ssd_iter_140000.caffemodel and Recognition with custom trained NN with ICT human dataset runing on OpenCV DNN")

    # imgLogo = tki.PhotoImage(file = os.path.sep.join([self.root_path, "ui/images/ailogoflat.png"]))
    # self.logo = tki.Label(self.bottom_status_frame,image=imgLogo)
    # self.logo.image = imgLogo
    # self.logo.pack(side="left", padx=0, pady=0)

    self.top_status_frame = tki.Frame(self.root)
    self.top_status_frame.pack(side=tki.TOP, fill=tki.X)
    self.top_status_frame.configure(bg='white')

    # img = tki.PhotoImage(file=os.path.sep.join([self.root_path, "ui/images/system_ready.png"]))
    # self.panel_pass = tki.Label(self.top_status_frame,image=img)
    # self.panel_pass.image = img
    # self.panel_pass.pack(side="left", padx=0, pady=0)

    self.minimize_button = tki.Button(self.top_status_frame, text="Minimize", height = 2, width = 6, command = self.minimize_window, font = ('Comic Sans MS',15),
    bg = 'white', fg = 'black', anchor='center')
    self.minimize_button.pack(side="right", padx=0, pady=0)

    self.top_label_cpu_text = tki.StringVar(self.root)
    self.top_label_cpu = tki.Label(self.top_status_frame, textvariable = self.top_label_cpu_text, height = 2, width = 4, font = ('Comic Sans MS',16),
    bg = 'white', fg = 'SkyBlue4', anchor='w')
    self.top_label_cpu.pack(side="right", padx=0, pady=0)
    self.top_label_cpu_text.set("VPU")

    self.top_label_text = tki.StringVar(self.root)
    self.top_label = tki.Label(self.top_status_frame, textvariable = self.top_label_text, height = 2, font = ('Comic Sans MS',16),
    bg = 'white', fg = 'SkyBlue4', anchor='w')
    self.top_label.pack(side="right", fill=tki.X, expand="yes", padx=0, pady=0)
    self.top_label_text.set("System Ready")



    self.video_frame = tki.Frame(self.root)
    self.video_frame.pack(side=tki.LEFT)
    self.video_frame.configure(bg='white')
    self.pb = ttk.Progressbar(self.video_frame, orient='horizontal', mode='indeterminate')
