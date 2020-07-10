import numpy as np
import cv2
import utils
from facial_landmarks_detection import FacialLandmarksDetection

COLOR = (0,0,255)
model_path = "../models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
device_name = 'CPU'
image_path = "../bin/face6_cropped.jpg"

landmarks_model = FacialLandmarksDetection(model_path)
landmarks_model.load_model(device_name)
image = cv2.imread(image_path)
input_height, input_width, input_channels = image.shape
processed_image=landmarks_model.preprocess_input(image)
request_handle=landmarks_model.predict(processed_image,0)
request_handle.wait()
output=landmarks_model.get_output(request_handle)
eye_boxes=landmarks_model.preprocess_output(output, input_width, input_height)
cropped_eyes=utils.crop_image(image,eye_boxes)
cv2.imshow('eye1',cropped_eyes[0])
cv2.imshow('eye2',cropped_eyes[1])
cv2.imwrite('../bin/face6_left_eye.jpg',cropped_eyes[0])
cv2.imwrite('../bin/face6_right_eye.jpg',cropped_eyes[1])
cv2.waitKey(0)