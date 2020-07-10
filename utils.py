import numpy as np
import cv2
import sys
import logging as log

PADDING_COLOR = [0,0,0]
CALIBRATION_FILE = 'calibration.npz'

COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255),(128,0,0),(0,128,0),(0,0,128)]
LINE_THICKNESS = 1
FONT_THICKNESS = 1
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
BOX_SIDE_LENGTH = 80

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

def imshow_fullscreen (winname, img):
	cv2.namedWindow (winname, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty (winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.imshow (winname, img)

def get_calibration(cal_points = None):
    #get calibration values that will be used to scale model output to screen dimensions
    #if no calibration points dictionary is passed in, it will be retrieved from file
    if cal_points is None:
    	try:
    		cal_points = np.load(CALIBRATION_FILE)
    	except IOError:
    		log.critical("ERROR: Couldn't open calibration file, use --calibrate first")
    		exit()

    xmin = (cal_points['top_left'][0] + cal_points['bottom_left'][0])/2
    xmax = (cal_points['top_right'][0] + cal_points['bottom_right'][0])/2
    ymin = (cal_points['top_left'][1] + cal_points['top_right'][1])/2
    ymax = (cal_points['bottom_left'][1] + cal_points['bottom_right'][1])/2
    return [xmin, xmax], [ymin, ymax]

def save_calibration(cal_points):
	np.savez(CALIBRATION_FILE, 		top_left=cal_points['top_left'], 
									top_right=cal_points['top_right'], 
									bottom_left=cal_points['bottom_left'], 
									bottom_right=cal_points['bottom_right'])


def display_inference_results(frame, face_boxes, head_pose, gaze_vector, inference_time):
	
	cv2.namedWindow ('window', cv2.WINDOW_NORMAL)
	cv2.setWindowProperty ('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	img = np.copy(frame)

	#Highlight detected face and eyes
	cv2.rectangle(img, face_boxes[0]['pt1'], face_boxes[0]['pt2'],COLORS[0],LINE_THICKNESS)

	#flip horizontally to match cursor position
	flipped_img=cv2.flip(img,1)

	#Show head pose output:
	cv2.putText(flipped_img, 
		"Head pose: Roll = {:.1f}, Pitch = {:.1f} Yaw = {:.1f}".format(head_pose['roll'],head_pose['pitch'],head_pose['yaw']), 
		(10,20), 
		FONT, 
		FONT_SCALE, 
		COLORS[2], 
		FONT_THICKNESS)

	#Show gaze estimation output:
	cv2.putText(flipped_img, 
		"Gaze vector: ({:.1f}, {:.1f}, {:.1f})".format(gaze_vector[0],gaze_vector[1],gaze_vector[2]), 
		(10,40), 
		FONT, 
		FONT_SCALE, 
		COLORS[2], 
		FONT_THICKNESS)
		
	#Show total inference time:
	cv2.putText(flipped_img, 
		"Total inference time: {:d} msec".format(int(inference_time*1000)), 
		(10,60), 
		FONT, 
		FONT_SCALE, 
		COLORS[2], 
		FONT_THICKNESS)
	

	cv2.imshow('window',flipped_img)
