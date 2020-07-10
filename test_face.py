import numpy as np
import cv2
import utils
from face_detection import FaceDetection


model_path = "../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml"
device_name = 'CPU'
image_path = "../bin/face6.jpg"
#image_path = "../bin/Datasets/HeadPose/Person01/person01101-60-90.jpg"
threshold = 0.8


face_network = FaceDetection(model_path)
face_network.load_model(device_name)
image = cv2.imread(image_path)
input_height, input_width, input_channels = image.shape
processed_image=face_network.preprocess_input(image)
request_handle=face_network.predict(processed_image,0)
request_handle.wait()
output=face_network.get_output(request_handle)
boxes=face_network.preprocess_output(output,threshold,input_width,input_height)
crop_img=utils.crop_image(image,boxes)
print(output)
cv2.imshow('img',crop_img[0])
cv2.imwrite('../bin/face6_cropped.jpg',crop_img[0])
cv2.waitKey(0)
