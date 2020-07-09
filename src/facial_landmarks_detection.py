from model import Model
import utils
import numpy as np

class FacialLandmarksDetection(Model):
    """
    Load and configure inference plugins for the specified target devices 
    and perform asynchronous infer requests.
    Written for OpenVino's pretrained model: landmarks-regression-retail-0009
    """

    INPUT_HEIGHT = 48
    INPUT_WIDTH = 48
    EYE_RADIUS_RATIO = 10

    def get_output(self, request_handle):
        output=np.squeeze(request_handle.outputs["95"])
        return output

    def preprocess_input(self, image):
        return utils.resize_image(image, FacialLandmarksDetection.INPUT_HEIGHT, FacialLandmarksDetection.INPUT_WIDTH)

    def preprocess_output(self, output, threshold, img_width, img_height):
        eye1_pos = (int(output[0]*img_width), int(output[1]*img_height))
        eye2_pos = (int(output[2]*img_width), int(output[3]*img_height))
        eye_radius = img_height // FacialLandmarksDetection.EYE_RADIUS_RATIO 
        eye1_box = {'pt1':(eye1_pos[0]-eye_radius,eye1_pos[1]-eye_radius), 'pt2':(eye1_pos[0]+eye_radius,eye1_pos[1]+eye_radius)}
        eye2_box = {'pt1':(eye2_pos[0]-eye_radius,eye2_pos[1]-eye_radius), 'pt2':(eye2_pos[0]+eye_radius,eye2_pos[1]+eye_radius)}
        eyes=[eye1_box, eye2_box]
        return eyes
