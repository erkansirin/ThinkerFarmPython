import os

root_path = os.path.expanduser('/Volumes/DL/edge')
protoPath = os.path.sep.join([root_path,"/models/face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join([root_path,"/models/face_detection_model",
    "res10_300x300_ssd_iter_140000.caffemodel"])
recognizer_modelPath = os.path.sep.join([root_path,"/models/face_recognition_models/recognizer.pickle"])
recognizer_le = os.path.sep.join([root_path,"/models/face_recognition_models/le.pickle"])
dlib_encodings = os.path.sep.join([root_path,"/models/face_recognition_models/encodings.pickle"])
torch_model = os.path.sep.join([root_path,"/models/face_detection_model","openface_nn4.small2.v1.t7"])
