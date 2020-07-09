from model import Model
import utils
import numpy as np

class FaceDetection(Model):
    """
    Load and configure inference plugins for the specified target devices 
    and perform asynchronous infer requests.
    Written for OpenVino's pretrained model: face-detection-adas-binary-0001
    """
    INPUT_HEIGHT = 384
    INPUT_WIDTH = 672

    def get_output(self, request_handle):
        output=np.squeeze(request_handle.outputs["detection_out"])
        return output

    def preprocess_input(self, image):
        return utils.resize_image(image,FaceDetection.INPUT_HEIGHT, FaceDetection.INPUT_WIDTH)

    def preprocess_output(self, output, threshold, img_width, img_height):
        face_detections = output[output[:,1]==1]
        face_detections = face_detections[face_detections[:,2]>=threshold]
        boxes=[]
        for detection in face_detections:
            pt1=(int(detection[3]*img_width), int(detection[4]*img_height))
            pt2=(int(detection[5]*img_width), int(detection[6]*img_height))
            box = {'pt1':pt1 , 'pt2':pt2}
            boxes.append(box)
        return boxes
