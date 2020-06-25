import numpy as np
import cv2
import sys

PADDING_COLOR = [0,0,0]

def crop_image(image, boxes):
	'''
	Returns:
		0 if no face was detected
		-1 if more than one face was detected
		cropped face if exactly one face was detected
	Adds padding to make the processed image square, this prevents distortion when feeding to next model
	'''
	if len(boxes)==0:
		return 0
	elif len(boxes)>1:
		return -1
	else:
		x1=boxes[0]['pt1'][0]
		x2=boxes[0]['pt2'][0]
		y1=boxes[0]['pt1'][1]
		y2=boxes[0]['pt2'][1]
		crop_img = image[y1:y2, x1:x2]
	
	box_height=y2-y1
	box_width=x2-x1
	output_side_length = max(box_width, box_height)
	
	pad_x=(output_side_length-box_width)//2
	pad_y=(output_side_length-box_height)//2

	processed_img= cv2.copyMakeBorder(crop_img,pad_y,pad_y,pad_x,pad_x,cv2.BORDER_CONSTANT,value=PADDING_COLOR)
	return processed_img