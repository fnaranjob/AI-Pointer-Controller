import numpy as np
import cv2
import utils
from head_pose_estimation import HeadPoseEstimation

model_path = "../models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml"
device_name = 'CPU'
image_path = "../bin/face6_cropped.jpg"

head_pose_network = HeadPoseEstimation(model_path)
head_pose_network.load_model(device_name)
image = cv2.imread(image_path)
input_height, input_width, input_channels = image.shape
processed_image=head_pose_network.preprocess_input(image)
request_handle=head_pose_network.predict(processed_image,0)
request_handle.wait()
output=head_pose_network.get_output(request_handle)
processed_output=head_pose_network.preprocess_output(output)
print(processed_output)
cv2.imshow('img',image)
cv2.waitKey(0)