# definitions.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# definitions.py create for the sake of smaller
# main app file main_application.py import this class and use objects in it


import cv2
from utilities.capture_threading import capture_threading

def setup_definitions(self, vs, args):
    self.frame = None
    self.thread = None
    self.stop_event = None
    self.panel = None
    self.update_final_text = 0
    self.active_menu = -2
    self.vs = vs
    self.active_training = False
    #self.vcap = cv2.VideoCapture("rtsp://admin:1803Eg15@192.168.2.112:554/onvif1")
    #ßæself.vcap = capture_threading("rtsp://admin:1803Eg15@192.168.2.112:554/onvif1")

    #public working

    #self.vcap = capture_threading("http://79.108.129.171:9000/mjpg/video.mjpg")
    #self.vcap = capture_threading("http://213.226.254.135:91/mjpg/video.mjpg")#çek cumhuriyeti
    self.vcap = capture_threading("http://root:admin@10.0.30.236/mjpg/video.mjpg") #Serbia novi pazar
    #self.vcap = capture_threading("http://93.41.227.105:81/GetData.cgi?CH=1") #Lombardia, Varese






    self.vcap.start()

    self.cpu_type = args["target"]
    #self.root_path = args["path"]
    self.current_id = -1
    self.selected_id = 0
    self.take_allowed = 0
    self.total_scanned_face = 10
    self.countdown_time = 10
    self.real_countdown_time = 3
    self.real_countdown_time_train = 3



    self.ssd_caffe_confidence = 0.2
    self.face_confidence = 0.35
    self.classifier_confidence = 0.15

    self.menu_link = ""
    self.count_time = 1
    self.pose_dataset = "COCO"
    self.pose_confidence = 0.1
