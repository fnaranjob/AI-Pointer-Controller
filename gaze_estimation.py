from model import Model
import utils
import numpy as np

class GazeEstimation(Model):
    """
    Load and configure inference plugins for the specified target devices 
    and perform asynchronous infer requests.
    Written for OpenVino's pretrained model: gaze-estimation-adas-0002
    """

    INPUT_HEIGHT = 60
    INPUT_WIDTH = 60
    INPUT1_NAME = 'left_eye_image'
    INPUT2_NAME = 'right_eye_image'
    INPUT3_NAME = 'head_pose_angles'

    def predict(self, left_eye_img, right_eye_img, head_angles, req_id):
        input3=np.array([head_angles]) 
        input_dict={GazeEstimation.INPUT1_NAME:left_eye_img, GazeEstimation.INPUT2_NAME:right_eye_img, GazeEstimation.INPUT3_NAME:input3}
        request_handle=self.exec_net.start_async(request_id=req_id, inputs=input_dict)
        return request_handle

    def get_output(self, request_handle):
        output=np.squeeze(request_handle.outputs["gaze_vector"])
        return output

    def preprocess_input(self, image):
        return utils.resize_image(image, GazeEstimation.INPUT_HEIGHT, GazeEstimation.INPUT_WIDTH)