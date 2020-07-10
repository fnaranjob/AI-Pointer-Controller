import numpy as np
import logging as log
import cv2
import pyautogui
import configargparse
import time

import utils
from input_feeder import InputFeeder 
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation

#CONSTANTS
FACE_DETECTION_THRESHOLD = 0.8
FILTER_QUANTITY = 1 #FILTER_QUANTITY samples will be averaged to stabilize mouse position
MOUSE_MOVE_TIME = 0.0 #Seconds
SCREEN_WIDTH = pyautogui.size().width
SCREEN_HEIGHT = pyautogui.size().height
SCREEN_X_LIMITS = [10.0, SCREEN_WIDTH-10]
SCREEN_Y_LIMITS = [10, SCREEN_HEIGHT-10]
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255),(128,0,0),(0,128,0),(0,0,128)]
LINE_THICKNESS = -1 #filled
FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
BOX_SIDE_LENGTH = 80

#PYAUTOGUI SETUP
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = configargparse.ArgumentParser("AI gaze mouse pointer controller")
    parser.add("-c", "--config_file", is_config_file=True,
                        help="Path to config file")
    parser.add("-t", "--input_type", required=True, type=str, default='cam',
                        help="Input type, use 'cam' for camera, 'image' for single image, 'video' for video file")
    parser.add("-p", "--input_path", type=str, default=None, 
    					help="Path to input, unused if input_type='cam'")
    parser.add("--face_detection_model", type=str, required=True,
                        help="Path to face detection model xml")
    parser.add("--head_pose_model", type=str, required=True,
                        help="Path to head pose estimation model xml")
    parser.add("--facial_landmarks_model", type=str, required=True,
                        help="Path to facial landmarks detection model xml")
    parser.add("--gaze_estimation_model", type=str, required=True,
                        help="Path to gaze estimation model xml")
    parser.add("-d", "--device", type=str, default='CPU',
                        help="Device to run inference on")
    parser.add("--calibrate", action='store_true',
                        help="Run camera calibration")
    parser.add("--display_all", action='store_true',
                        help="Display all models' outputs")
    return parser

def get_screen_position(x, y, cal_x_limits, cal_y_limits):
    screen_x = utils.rescale(x, cal_x_limits, SCREEN_X_LIMITS)
    screen_y = utils.rescale(y, cal_y_limits, SCREEN_Y_LIMITS)
    return screen_x, screen_y

def get_base_img(line1, line2, color):
    base_img = np.zeros([SCREEN_HEIGHT,SCREEN_WIDTH,3],dtype=np.uint8)
    base_img.fill(255)
    line1_size, _ = cv2.getTextSize(line1,FONT,FONT_SCALE,FONT_THICKNESS)
    line2_size, _ = cv2.getTextSize(line2,FONT,FONT_SCALE,FONT_THICKNESS)
    line1_x = (SCREEN_WIDTH - line1_size[0])//2
    line2_x = (SCREEN_WIDTH - line2_size[0])//2
    cv2.putText(base_img, line1, (line1_x,SCREEN_HEIGHT//2-20), FONT, FONT_SCALE, color, FONT_THICKNESS)
    cv2.putText(base_img, line2, (line2_x,SCREEN_HEIGHT//2+20), FONT, FONT_SCALE, color, FONT_THICKNESS)
    return base_img

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
    return np.array(output)


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
    gaze_vector_accum = np.array([0,0,0],dtype='float64')
    gaze_vector_filtered = np.array([0,0,0],dtype='float64')
    
    #get screen calibration
    if not args.calibrate:
        run_calibration = False
        cal_x_limits, cal_y_limits = utils.get_calibration()
    else:
        run_calibration = True
        update_display = True
        
        #squares to draw on screen for calibration
        top_left_square = {'pt1':(0,0), 'pt2':(BOX_SIDE_LENGTH,BOX_SIDE_LENGTH)}
        top_right_square = {'pt1':(SCREEN_WIDTH - BOX_SIDE_LENGTH,0), 'pt2':(SCREEN_WIDTH, BOX_SIDE_LENGTH)}
        bottom_left_square = {'pt1':(0,SCREEN_HEIGHT - BOX_SIDE_LENGTH), 'pt2':(BOX_SIDE_LENGTH,SCREEN_HEIGHT)}
        bottom_right_square =   {'pt1':(SCREEN_WIDTH - BOX_SIDE_LENGTH,SCREEN_HEIGHT - BOX_SIDE_LENGTH), 
                                'pt2':(SCREEN_WIDTH,SCREEN_HEIGHT)}
        cal_squares = [top_left_square,top_right_square,bottom_left_square, bottom_right_square]
        
        #names of the calibration points for storing on calibration file
        cal_names = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        
        #model output values for each calibration point will be stored here
        cal_points = {}

        square_iter = iter(cal_squares)
        name_iter = iter(cal_names)
        
        #image to display on screen for calibration
        base_img = get_base_img("LOOK AT THE SQUARES FOR 2 SECONDS","AND THEN PRESS n", COLORS[0])
        

    if not single_image_mode:
        while True:

            #filter results
            count += 1
            if(count>FILTER_QUANTITY):
                gaze_vector_filtered=gaze_vector_accum/FILTER_QUANTITY
                gaze_vector_accum=np.array([0,0,0],dtype='float64')
                count=0

            #process frames
            frame = next(input_feed.next_batch())
            
            start_time=time.time()
            face_boxes = run_inference(frame, face_model)
            cropped_faces = utils.crop_image(frame,face_boxes)
            
            if cropped_faces==0: #no face detected, nothing to process
                continue

            elif cropped_faces is None: #finished reading input feed
                break

            elif len(cropped_faces)==1: #found a single face in the frame, proceed
                
                head_pose = run_inference(cropped_faces[0], head_pose_model)
                eye_boxes = run_inference(cropped_faces[0], facial_landmarks_model)
                cropped_eyes = utils.crop_image(cropped_faces[0], eye_boxes)
                gaze_vector = run_inference_gaze(cropped_eyes[0], cropped_eyes[1], head_pose, gaze_estimation_model)
                
                inference_time=time.time()-start_time
                
                gaze_vector_accum += gaze_vector
                
                if run_calibration:
                    
                    if update_display:
                        img = np.copy(base_img)
                        square = next(square_iter, None)
                        if not square is None: 
                            cv2.rectangle(img,square['pt1'], square['pt2'],COLORS[0],-1)
                            update_display=False
                        else: #Done with calibration
                            cal_x_limits, cal_y_limits = utils.get_calibration(cal_points)
                            utils.save_calibration(cal_points)
                            run_calibration=False

                    utils.imshow_fullscreen('window',img)

                    if cv2.waitKey(1) & 0xFF == ord('n'):
                        update_display = True
                        point = np.array([ gaze_vector_filtered[0], gaze_vector_filtered[1] ])
                        point_name = next(name_iter)
                        cal_points[point_name] = point
                    
                else:
                    
                    if not args.display_all:
                        img = get_base_img("GAZE CONTROL ENABLED", "MOVE MOUSE TO ANY CORNER OR PRESS q TO EXIT", COLORS[1])
                        utils.imshow_fullscreen('window',img)
                    else:
                        utils.display_inference_results(frame, face_boxes, head_pose, gaze_vector, inference_time)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User terminated program, goodbye")
                        break
                    

                    screen_x, screen_y = get_screen_position(gaze_vector_filtered[0], gaze_vector_filtered[1], cal_x_limits, cal_y_limits)
                
                    try:
                        pyautogui.moveTo(screen_x,screen_y,MOUSE_MOVE_TIME)
                    except pyautogui.FailSafeException:
                        print("User terminated program, goodbye")
                        break

            else:
                #Handle multiple people here if needed
                log.critical("ERROR: Multiple people detected")
                break
            
    else:
        #Implement single image mode here if needed
        log.critical("ERROR: Single image mode not implemented")

    input_feed.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()