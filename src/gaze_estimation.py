import numpy as np
import cv2
import sys
from openvino.inference_engine import IECore


class GazeEstimation:
    """
    Load and configure inference plugins for the specified target devices 
    and perform asynchronous infer requests.
    Written for OpenVino's pretrained model: gaze-estimation-adas-0002
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
        pass
        #input_name = next(iter(self.net.inputs))
        #input_dict={input_name:image}
        #request_handle=self.exec_net.start_async(request_id=req_id, inputs=input_dict)
        #return request_handle

    def get_output(self, request_handle):
        pass
        #output=np.squeeze(request_handle.outputs["95"])
        #return output

    def preprocess_input(self, image):
        pass
        #processed_image = np.copy(image)
        #processed_image = cv2.resize(processed_image,(FacialLandmarksDetection.INPUT_WIDTH,FacialLandmarksDetection.INPUT_HEIGHT))
        #processed_image = processed_image.transpose((2,0,1))
        #processed_image = processed_image.reshape(1, 3, FacialLandmarksDetection.INPUT_HEIGHT, FacialLandmarksDetection.INPUT_WIDTH)
        #return processed_image

    def preprocess_output(self, output, img_width, img_height):
        pass
        #eye1_pos = (int(output[0]*img_width), int(output[1]*img_height))
        #eye2_pos = (int(output[2]*img_width), int(output[3]*img_height))
        #eye_radius = img_height // FacialLandmarksDetection.EYE_RADIUS_RATIO 
        #eye1_box = {'pt1':(eye1_pos[0]-eye_radius,eye1_pos[1]-eye_radius), 'pt2':(eye1_pos[0]+eye_radius,eye1_pos[1]+eye_radius)}
        #eye2_box = {'pt1':(eye2_pos[0]-eye_radius,eye2_pos[1]-eye_radius), 'pt2':(eye2_pos[0]+eye_radius,eye2_pos[1]+eye_radius)}
        #eyes=[eye1_box, eye2_box]
        #eturn eyes