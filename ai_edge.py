# ai_edge.py
#
#
# Author Erkan SIRIN
# Created for AI Edge project.
#
# ai_edge.py contains init arguments live video stream object from imutils.video
# and main application loop


from __future__ import print_function
from main_application import main_application
from imutils.video import VideoStream
import argparse
import time

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=14, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

ap = argparse.ArgumentParser()
ap.add_argument("-rp", "--path", required=False,
	help="path to project root")
ap.add_argument("-p", "--cameratype", type=int, default=-1,
	help="0 is defualt device camera laptop, desktop etc. 1 is Raspberry Pi camera 2 is Jetson Nano camera module")
ap.add_argument("-t", "--target", type=int, default=0,
	help="cpu or intel vpu target selection default 0 is sytem CPU 1 is Intel VPU")
args = vars(ap.parse_args())

print("AI Edge : Opening camera")

target = args["target"]
if target == 2:
	vs = VideoStream(gstreamer_pipeline(flip_method=2))
else:
	vs = VideoStream(usePiCamera=args["cameratype"] > 0)
time.sleep(1.0)

pba = main_application(vs, args)
# pba.root.overrideredirect(0)
#pba.root.wm_attributes('-type', 'splash')
# pba.root.wm_attributes('-fullscreen','false')
# pba.root.config(cursor="none")
pba.root.mainloop()
