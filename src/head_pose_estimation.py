from model import Model
import utils
import numpy as np


class HeadPoseEstimation(Model):
    """
    Load and configure inference plugins for the specified target devices 
    and perform asynchronous infer requests.
    Written for OpenVino's pretrained model: head-pose-estimation-adas-0001
    """

    INPUT_HEIGHT = 60
    INPUT_WIDTH = 60

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
