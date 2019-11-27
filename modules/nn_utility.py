import cv2



class nn_utility:

    def __init__(self,cpu_type):
        print("class init")

        self.net_detector = None
        self.cpu_type = cpu_type

    def setup_network(self, net_config_file_path, net_weight_file_path, framework):

        try:
            if framework == "Torch":
                self.net_detector = cv2.dnn.readNet(net_weight_file_path, framework)
                if self.cpu_type == 1:
                    self.net_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
                if self.cpu_type == 2:
                    self.net_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            if framework == "caffe":
                self.net_detector = cv2.dnn.readNet(net_config_file_path, net_weight_file_path, framework)
                if self.cpu_type == 1:
                    self.net_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
                if self.cpu_type == 2:
                    self.net_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        except RuntimeError as e:
            print("can not create net_detector :",e)



    def nn_detector(self,image_blob):

        self.net_detector.setInput(image_blob)
        detections = self.net_detector.forward()

        return detections
