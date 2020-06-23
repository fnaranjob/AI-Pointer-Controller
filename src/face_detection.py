import numpy as np
import sys
from openvino.inference_engine import IECore


class FaceDetection:
    """
    Load and configure inference plugins for the specified target devices 
    and perform asynchronous infer requests.
    Written for OpenVino's pretrained model: face-detection-adas-binary-0001
    """
    IE=None
    net=None
    exec_net=None
    device=None

    def __init__(self, model_xml, extensions=None):
        self.IE=IECore()
        self.net=IENetwork(model=model_xml,weights=model_xml.replace('xml','bin'))

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

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
