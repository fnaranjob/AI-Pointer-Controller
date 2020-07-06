import numpy as np
import logging as log
import cv2
import pyautogui
from argparse import ArgumentParser

import utils
from input_feeder import InputFeeder 
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation

#CONSTANTS
FACE_DETECTION_THRESHOLD = 0.8
FILTER_QUANTITY = 10
CALIBRATION_FILE = 'calibration.npz'
SCREEN_X_LIMITS = [20.0, 1900.0]
SCREEN_Y_LIMITS = [20.0, 1060.0]
MOUSE_MOVE_TIME = 0.0 #Seconds

#PYAUTOGUI SETUP
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

 
def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser("AI gaze mouse pointer controller")
    parser.add_argument("-t", "--input_type", required=True, type=str,
                        help="Input type, use 'cam' for camera, 'image' for single image, 'video' for video file")
    parser.add_argument("-p", "--input_path", type=str, default=None, 
    					help="Path to input, unused if input_type='cam'")
    parser.add_argument("--face_detection_model", type=str, required = True,
                        help="Path to face detection model xml")
    parser.add_argument("--head_pose_model", type=str, required=True,
                        help="Path to head pose estimation model xml")
    parser.add_argument("--facial_landmarks_model", type=str, required=True,
                        help="Path to facial landmarks detection model xml")
    parser.add_argument("--gaze_estimation_model", type=str, required=True,
                        help="Path to gaze estimation model xml")
    parser.add_argument("-d", "--device", type=str, default='CPU',
                        help="Device to run inference on")
    return parser

def get_calibration():
    #get calibration values that will be used to scale model output to screen dimensions
    cal_points = np.load(CALIBRATION_FILE)
    xmin = (cal_points['top_left'][0] + cal_points['bottom_left'][0])/2
    xmax = (cal_points['top_right'][0] + cal_points['bottom_right'][0])/2
    ymin = (cal_points['top_left'][1] + cal_points['top_right'][1])/2
    ymax = (cal_points['bottom_left'][1] + cal_points['bottom_right'][1])/2
    return [xmin, xmax], [ymin, ymax]

def get_screen_position(x, y, cal_x_limits, cal_y_limits):
    screen_x = utils.rescale(x, cal_x_limits, SCREEN_X_LIMITS)
    screen_y = utils.rescale(y, cal_y_limits, SCREEN_Y_LIMITS)
    return screen_x, screen_y

def run_inference(frame, model):
    if frame is None:
        return None
    input_height, input_width, _ = frame.shape
    processed_frame=model.preprocess_input(frame)
    request_handle=model.predict(processed_frame,req_id=0)
    request_handle.wait()
    output=model.get_output(request_handle)

    #not all models use all arguments passed to this method, some will be left unused, 
    #this was done deliberately to avoid repeating code
    processed_output=model.preprocess_output(output=output, 
                                            threshold=FACE_DETECTION_THRESHOLD, 
                                            img_width=input_width, 
                                            img_height=input_height)
    return processed_output

def run_inference_gaze(left_eye, right_eye, angles, model):
    left_eye_processed = model.preprocess_input(left_eye)
    right_eye_processed = model.preprocess_input(right_eye)
    angles_list = [angles['yaw'], angles['pitch'], angles['roll']]
    request_handle = model.predict(left_eye_processed, right_eye_processed, angles_list, req_id=0)
    request_handle.wait()
    output = model.get_output(request_handle)
    return output


def main():
    args = build_argparser().parse_args()
    single_image_mode = (args.input_type == 'image') 

    #Create and validate input feed
    input_feed = InputFeeder(input_type=args.input_type,input_file=args.input_path)
    input_feed.load_data()

    if not input_feed.is_open():
        log.critical('Error opening input, check --input_path parameter (use --help for more info)')
        exit()

    #Load models
    face_model = FaceDetection(args.face_detection_model)
    face_model.load_model(args.device)
    head_pose_model = HeadPoseEstimation(args.head_pose_model)
    head_pose_model.load_model(args.device)
    facial_landmarks_model = FacialLandmarksDetection(args.facial_landmarks_model)
    facial_landmarks_model.load_model(args.device)
    gaze_estimation_model = GazeEstimation(args.gaze_estimation_model)
    gaze_estimation_model.load_model(args.device)

    #initialize frame count for filtering
    count = 0
    gaze_vector_accum = [0,0,0]
    gaze_vector_filtered = [0,0,0]

    #get screen calibration
    cal_x_limits, cal_y_limits = get_calibration()

    if not single_image_mode:
        while True:

            #filter results
            count += 1
            if(count>FILTER_QUANTITY):
                gaze_vector_filtered=gaze_vector_accum/FILTER_QUANTITY
                gaze_vector_accum=[0,0,0]
                count=0

            #process frames
            frame = next(input_feed.next_batch())
            
            face_boxes = run_inference(frame, face_model)
            cropped_faces = utils.crop_image(frame,face_boxes)
            
            if cropped_faces==0: #no face detected, nothing to process
                continue
            elif cropped_faces is None: #finished reading input feed
                break
            elif len(cropped_faces)==1:
                head_pose = run_inference(cropped_faces[0], head_pose_model)
                eye_boxes = run_inference(cropped_faces[0], facial_landmarks_model)
                cropped_eyes = utils.crop_image(cropped_faces[0], eye_boxes)
                gaze_vector = run_inference_gaze(cropped_eyes[0], cropped_eyes[1], head_pose, gaze_estimation_model)
                gaze_vector_accum += gaze_vector
                screen_x, screen_y = get_screen_position(gaze_vector_filtered[0], gaze_vector_filtered[1], cal_x_limits, cal_y_limits)
                pyautogui.moveTo(screen_x,screen_y,MOUSE_MOVE_TIME)

                #debugging output
                #cv2.imshow('face',cropped_eyes[0])
                #print('(',gaze_vector_filtered[0],' , ',gaze_vector_filtered[1],')')
                #print('(',screen_x,' , ',screen_y,')')
            else:
                #TODO Handle multiple people
                pass
            
            #Quit app if q is pressed
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
    else:
        pass

    input_feed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()