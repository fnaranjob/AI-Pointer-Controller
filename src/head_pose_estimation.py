import numpy as np
import cv2
import sys
import utils
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
        output=request_handle.outputs
        return output

    def preprocess_input(self, image):
        return utils.resize_image(image, HeadPoseEstimation.INPUT_HEIGHT, HeadPoseEstimation.INPUT_WIDTH)

    def preprocess_output(self, output, threshold, img_width, img_height):
        roll=np.squeeze(output['angle_r_fc'])[()]
        pitch=np.squeeze(output['angle_p_fc'])[()]
        yaw=np.squeeze(output['angle_y_fc'])[()]
        output_dict={'roll':roll, 'pitch':pitch, 'yaw':yaw}
        return output_dict
