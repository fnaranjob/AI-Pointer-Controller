import numpy as np
import sys

def crop_image(image, boxes):
	'''
	Returns:
	0 if no face was detected
	-1 if more than one face was detected
	cropped face if exactly one face was detected
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
		return crop_img