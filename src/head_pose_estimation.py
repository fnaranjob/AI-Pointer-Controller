import numpy as np
import cv2
import sys
from openvino.inference_engine import IECore


class HeadPoseEstimation:
    """
    Load and configure inference plugins for the specified target devices 
    and perform asynchronous infer requests.
    Written for OpenVino's pretrained model: head-pose-estimation-adas-0001
    """
    IE=None
    net=None
    exec_net=None
    device=None
    INPUT_HEIGHT = 60
    INPUT_WIDTH = 60

    def __init__(self, model_xml):
        self.IE=IECore()
        self.net=self.IE.read_network(model=model_xml,weights=model_xml.replace('xml','bin'))

    def __check_layers__(self):
        layers_map = self.IE.query_network(network=self.net,device_name=self.device)
        for layer in self.net.layers.keys():
            if layers_map.get(layer, "none") == "none": #Found unsupported layer
                return False
        return True

    def load_model(self, device_name='CPU'):
        self.device=device_name
        if(self.__check_layers__()):
            self.exec_net=self.IE.load_network(network=self.net,device_name=device_name,num_requests=1)
        else:
            sys.exit("Unsupported layer found, can't continue")

    def predict(self, image, req_id):
        input_name = next(iter(self.net.inputs))
        input_dict={input_name:image}
        request_handle=self.exec_net.start_async(request_id=req_id, inputs=input_dict)
        return request_handle

    def get_output(self, request_handle):
        pass
        #output=np.squeeze(request_handle.outputs["detection_out"])
        #return output

    def preprocess_input(self, image):
        processed_image = np.copy(image)
        processed_image = cv2.resize(processed_image,(HeadPoseEstimation.INPUT_WIDTH,HeadPoseEstimation.INPUT_HEIGHT))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, 3, HeadPoseEstimation.INPUT_HEIGHT, HeadPoseEstimation.INPUT_WIDTH)
        return processed_image

    def preprocess_output(self, output, threshold, img_width, img_height):
        pass
        #face_detections = output[output[:,1]==1]
        #face_detections = face_detections[face_detections[:,2]>=threshold]
        #boxes=[]
        #for detection in face_detections:
        #    pt1=(int(detection[3]*img_width), int(detection[4]*img_height))
        #    pt2=(int(detection[5]*img_width), int(detection[6]*img_height))
        #    box = {'pt1':pt1 , 'pt2':pt2}
        #    boxes.append(box)
        #return boxes
