import numpy as np
import cv2
import utils
from gaze_estimation import GazeEstimation

model_path = "../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml"
device_name = 'CPU'
left_eye_path = "../bin/face2_left_eye.jpg"
right_eye_path = "../bin/face2_right_eye.jpg"

gaze_estimation_network = GazeEstimation(model_path)
gaze_estimation_network.load_model(device_name)
left_eye_image = cv2.imread(left_eye_path)
right_eye_image = cv2.imread(right_eye_path)
input_height, input_width, input_channels = left_eye_image.shape
left_eye_processed = gaze_estimation_network.preprocess_input(left_eye_image)
right_eye_processed = gaze_estimation_network.preprocess_input(right_eye_image)
request_handle=gaze_estimation_network.predict(left_eye_processed, right_eye_processed,[10,10,10],0)
request_handle.wait()
output=gaze_estimation_network.get_output(request_handle)
#processed_output=head_pose_network.preprocess_output(output)
print(np.linalg.norm(output))
cv2.imshow('left eye',left_eye_image)
cv2.imshow('right eye',right_eye_image)
cv2.waitKey(0)