import numpy as np
import cv2
import sys

PADDING_COLOR = [0,0,0]

def rescale(value, input_range, output_range):
	slope = (output_range[1] - output_range[0]) / (input_range[1] - input_range[0])
	output = output_range[0] + slope * (value - input_range[0])
	if output > output_range[1]:
		output = output_range[1]
	elif output < output_range[0]:
		output = output_range[0]

	return output


def resize_image(image, height, width):
	processed_image = np.copy(image)
	processed_image = cv2.resize(processed_image,(width,height))
	processed_image = processed_image.transpose((2,0,1))
	processed_image = processed_image.reshape(1, 3, height, width)
	return processed_image

def crop_image(image, boxes):
	'''
	Returns:
		0 if boxes is empty
		cropped image (forced square aspect ratio) if not
	'''
	if len(boxes)==0:
		return 0
	else:
		cropped_images=[]
		for i,box in enumerate(boxes):
			x1=box['pt1'][0]
			x2=box['pt2'][0]
			y1=box['pt1'][1]
			y2=box['pt2'][1]
			box_height=y2-y1
			box_width=x2-x1
			output_side_length = max(box_width, box_height)
			pad_x=(output_side_length-box_width)//2
			pad_y=(output_side_length-box_height)//2
			cropped_images.append(image[y1-pad_y:y2+pad_y, x1-pad_x:x2+pad_x])

	return cropped_images