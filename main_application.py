# main_application.py
#
#
# Author Erkan SIRIN
# Created for ThinkerFarm Edge project.
#
# main_application.py is main application structure class

from main_imports import *
from ui.init_ui import *

import dlib.cuda as cuda



os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

class main_application:

    def __init__(self, vs, args):

        setup_definitions(self, vs, args)

        init_ui(self)
        self.open_main_view()

        print("self.cpu_type : ",self.cpu_type)

        if self.cpu_type == 0:
            self.top_label_cpu_text.set("CPU")
        if self.cpu_type == 1:
            self.top_label_cpu_text.set("VPU")
        if self.cpu_type == 2:
            self.top_label_cpu_text.set("GPU")

        self.root.wm_title("ThinkerFarm")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)
        print("cuda list :",cuda.get_num_devices())

    def video_loop(self):
        try:
            while not self.stop_event.is_set():

                if self.active_menu == -2:
                    self.frame = self.vs.read()
                    self.frame = imutils.resize(self.frame, width=525,height=480)
                if self.active_menu == 0:

                    #self.frame = cv2.imread(os.path.sep.join([self.root_path,"/dataset/300x300.jpg"]))
            		#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



                    self.frame = self.vs.read()
                    self.frame = imutils.resize(self.frame, width=525,height=480)
                    imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(self.frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)
                    self.frame = run_face_pickle(self,self.frame,imageBlob)
                if self.active_menu == 1:
                    self.frame = self.vs.read()
                    self.frame = imutils.resize(self.frame, width=525,height=480)
                    self.frame  = run_face_scan(self,self.frame)
                if self.active_menu == 2:
                    self.frame = self.vs.read()
                    self.frame = imutils.resize(self.frame, width=525,height=480)
                    self.frame = run_face_dlib(self,self.frame)
                if self.active_menu == 3:
                    self.frame = self.vs.read()
                    self.frame = run_ssd_yolo(self,self.frame)
                if self.active_menu == 4:
                    self.frame = self.vs.read()
                    self.frame = run_mobilenet_caffe(self,self.frame)
                if self.active_menu == 5:
                    self.frame = self.vs.read()
                    self.frame = run_ssd_face_cnn(self,self.frame)
                if self.active_menu == 6:
                    self.frame = self.vs.read()
                    self.frame = run_openpose_network(self,self.frame)
                if self.active_menu == 7:
                    self.frame = self.vs.read()
                    runOpenposeWithSSD(self,self.frame)
                if self.active_menu == 8:
                    self.frame = self.vs.read()
                    self.out_edge_frame = run_edge_network(self,self.frame)
                if self.active_menu == 9:

                    net, frame = self.vcap.read()
                    print("frame rtsp : ",frame)
                    #image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    #frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    self.frame = run_ssd_face_cnn_ipcam(self,frame)

                if self.active_menu == 10:
                    ret, frame = self.vcap.read()
                    run_openpose_on_ipcam(self,frame)
                if self.active_menu == 11:
                    self.frame = self.vs.read()
                    self.frame = run_faster_rcnn_tensorflow(self,self.frame)
                if self.active_menu == 12:
                    self.frame = self.vs.read()
                    self.frame = run_squeezenet_classification(self,self.frame)

                if self.active_menu == 13:
                    net, frame = self.vcap.read()
                    self.frame = run_face_dlib_ipcam(self,frame)

                if self.active_menu == 14:
                    net, frame = self.vcap.read()
                    self.frame = imutils.resize(frame, width=525,height=480)
                    imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(self.frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)
                    self.frame = run_face_pickle_ipcam(self,self.frame,imageBlob)
                if self.active_menu == 15:
                    self.frame = self.vs.read()
                    self.frame = imutils.resize(self.frame, width=525,height=480)
                    imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(self.frame, (525, 480)), 1.0, (525, 480),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)
                    self.frame = run_face_pickle_dlib(self,self.frame,imageBlob)
                if self.active_menu == 16:

                    #net, frameip = self.vcap.read()
                    self.frame = self.vs.read()
                    self.frame = imutils.resize(self.frame, width=1024,height=576)

                    run_face_dlib_twocam(self,self.frame)

                    #self.frame = frameip


                if self.active_menu == 8:
                    print("running menu 8")
                    image = Image.fromarray(self.out_edge_frame)
                    image = ImageTk.PhotoImage(image)
                else:
                    image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
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

        except RuntimeError as e:
            print("RuntimeError")

    def open_main_view(self):
        self.vs.start()
        print("ThinkerFarm : Main Menu Starting Video Stream")
        #time.sleep(5.0)

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

    def on_close(self):
        self.stop_event.set()
        self.vs.stop()
        self.root.quit()

    def call_home(self):
        self.open_main_view()

    def close_app(self):
        self.stop_event.set()
        self.vs.stop()
        self.root.quit()
        self.root.destroy()

    def minimize_window(self):
        self.root.wm_state("iconic")




    def start_face_dlib(self):

        self.active_menu = -1
        self.menu_link = "load_dlib_face_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_face_pickle(self):
        self.active_menu = -1
        self.menu_link = "load_pickle_face_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_face_pickle_ipcam(self):
        self.active_menu = -1
        self.menu_link = "start_face_pickle_ipcam"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_face_dlib_ssd(self):
        self.active_menu = -1
        self.menu_link = "run_face_pickle_dlib"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_face_dlib_ipcam(self):
        self.active_menu = -1
        self.menu_link = "start_face_dlib_ipcam"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()


    def start_face_twocam_noview(self):
        self.active_menu = "-"
        self.menu_link = "start_face_twocam_noview"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()



    def open_ssd_main(self):
        self.button_frame.pack_forget()
        self.ssd_button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def start_ssd_yolo(self):
        if self.cpu_type == 1:
            load_ssd_yolo_network(self)
            messagebox.showinfo("Info", "Intel Movidius is not supporting Darknet NN it will be slow")
        else:
            load_ssd_yolo_network(self)

    def start_mobilenet_caffe(self):
        self.active_menu = -1
        self.menu_link = "start_mobilenet_caffe"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_ssd_face_cnn(self):
        self.active_menu = 5

    def start_ssd_face_cnn_ipcam(self):
        self.active_menu = 9

    def back_from_ssd(self):
        print("resetMyriadDevice")
        #cv2.dnn.resetMyriadDevice()
        self.active_menu = -2


        self.ssd_button_frame.pack_forget()
        self.button_frame.pack(fill=tki.Y, side=tki.RIGHT)



    def back_from_pose_estimation(self):
        self.active_menu = -2
        self.posenet_button_frame.pack_forget()
        self.button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def open_pose_estimation(self):
        self.button_frame.pack_forget()
        self.posenet_button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def open_gans_main(self):
        print("open_gans_main")

    def callPoseTensor(self):
        print("call pose estimation")

    def start_posenet_coco(self):
        self.pose_dataset = 'COCO'
        self.active_menu = -1
        self.menu_link = "load_openpose_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_posenet_mpi(self):
        self.pose_dataset = 'MPI'
        self.active_menu = -1
        self.menu_link = "load_openpose_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_posenet_mpi_ipcam(self):
        self.active_menu = 10

    def start_posenet_body(self):
        self.pose_dataset = 'BODY25'
        self.active_menu = -1
        self.menu_link = "load_openpose_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_posenet_hand(self):
        self.pose_dataset = 'HAND'
        self.active_menu = -1
        self.menu_link = "load_openpose_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_posenet_mpi_with_ssd(self):
        self.pose_dataset = "MPI"
        self.active_menu = -1
        setupOpenPoseNetWithSSD(self)

    def start_edge_detection(self):
        self.active_menu = -1
        init_edge_network(self)

    def list_item_selected(self,evt):
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        self.selected_id = index

    def open_face_scan_menu(self):

        self.active_menu = -1

        self.menu_link = "load_scan_detector_face_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()


    def start_scan(self):

        self.current_id = self.selected_id
        self.thread_count = threading.Thread(target=self.countdown, args=())
        self.thread_count.start()






    def back_from_scan_menu(self):
        self.active_menu = -2
        self.scan_menu_button_frame.pack_forget()
        self.button_frame.pack(fill=tki.Y, side=tki.RIGHT)
        self.selected_id = -1
        self.current_id = -1
        self.total_scanned_face = 10



    def take_photo(self):
        self.take_allowed = 1

    def close_main_view(self):
        self.stop_event.set()

    def open_face_implementation_menu(self):
        self.button_frame.pack_forget()
        self.face_implementation_button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def back_from_face_implementations(self):
        self.active_menu = -2
        self.face_implementation_button_frame.pack_forget()
        self.button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def open_main_view(self):
        self.vs.start()
        print("ThinkerFarm : Starting Video Stream")
        time.sleep(5.0)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

    def open_training_menu(self):
        self.button_frame.pack_forget()
        self.training_button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def back_from_training(self):
        self.active_menu = -2
        self.training_button_frame.pack_forget()
        self.button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def start_face_training(self):
        self.active_menu = -1
        self.menu_link = "trainFace"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_face_training_dlib(self):
        self.active_menu = -1
        self.menu_link = "trainFace_dlib"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def open_image_classifier_main(self):
        self.button_frame.pack_forget()
        self.classifiers_button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def back_from_classifiers(self):
        self.active_menu = -2
        self.classifiers_button_frame.pack_forget()
        self.button_frame.pack(fill=tki.Y, side=tki.RIGHT)

    def start_faster_rcnn_tensorflow(self):
        self.active_menu = -1
        self.menu_link = "load_faster_rcnn_tensorflow_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()

    def start_squeezenet(self):
        self.active_menu = -1
        self.menu_link = "load_squeezenet_classification_network"
        self.thread_count = threading.Thread(target=self.countdown_menu, args=())
        self.thread_count.start()




    def countdown_menu(self) :
        self.real_countdown_time = self.count_time
        while self.real_countdown_time > 0:
            self.countdown_time = self.real_countdown_time
            self.real_countdown_time = self.real_countdown_time - 1
            time.sleep(1)
            if self.real_countdown_time == 0:
                if self.menu_link == "start_mobilenet_caffe":
                    load_mobilenet_caffe_network(self)
                if self.menu_link == "trainFace":
                    print("call train face")
                    extract_pickle_embeddings(self)
                if self.menu_link == "trainFace_dlib":
                    print("call train face")
                    download_dataset(self)
                if self.menu_link == "load_openpose_network":
                    load_openpose_network(self)
                if self.menu_link == "load_faster_rcnn_tensorflow_network":
                    load_faster_rcnn_tensorflow_network(self)
                if self.menu_link == "load_pickle_face_network":
                    load_pickle_face_network(self)
                if self.menu_link == "load_squeezenet_classification_network":
                    load_squeezenet_classification_network(self)
                if self.menu_link == "load_dlib_face_network":
                    load_dlib_face_network(self)
                if self.menu_link == "load_scan_detector_face_network":
                    load_scan_detector_face_network(self)
                    self.button_frame.pack_forget()
                    self.scan_menu_button_frame.pack(fill=tki.Y, side=tki.RIGHT)
                if self.menu_link == "start_face_pickle_ipcam":
                    load_pickle_face_network_ipcam(self)
                if self.menu_link == "start_face_dlib_ipcam":
                    load_dlib_face_network_ipcam(self)
                if self.menu_link == "run_face_pickle_dlib":
                    load_dlib_pickle_face_network(self)
                if self.menu_link == "start_face_twocam_noview":
                    load_dlib_face_network_twocam(self)



    def countdown(self) :
        self.real_countdown_time = 3
        while self.real_countdown_time > 0:
            self.countdown_time = self.real_countdown_time
            self.real_countdown_time = self.real_countdown_time - 1
            time.sleep(1)
            if self.real_countdown_time == 0:
                self.take_allowed = 1

    def countdown_train(self) :
        self.real_countdown_time_train = self.count_time
        while self.real_countdown_time_train > 0:
            self.real_countdown_time_train = self.real_countdown_time_train - 1
            time.sleep(1)
            if self.real_countdown_time_train == 0:

                now = datetime.datetime.now()
                date_time = now.strftime("%d-%m-%Y %H:%M:%S")


                header = {"accept": "*/*"}


                r = requests.get(url = 'http://139.162.142.162:1994/api/companies/1/training-mode', headers=header)
                data = r.json()
                #print("Train status : ",data["trainingMode"])
                self.active_training = data["trainingMode"]
                if self.active_training == True:
                    #print("start training")
                    self.active_menu = -1
                    self.menu_link = "trainFace_dlib"
                    self.thread_count = threading.Thread(target=self.countdown_menu, args=())
                    self.thread_count.start()

                else:
                    #print("stop training")
                    self.thread_count_training = threading.Thread(target=self.countdown_train, args=())
                    self.thread_count_training.start()
