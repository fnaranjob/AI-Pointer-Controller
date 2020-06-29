import numpy as np
import logging as log
import cv2
from argparse import ArgumentParser

import utils
from input_feeder import InputFeeder 
from face_detection import FaceDetection
import facial_landmarks_detection
import gaze_estimation
import head_pose_estimation

#CONSTANTS
FACE_DETECTION_THRESHOLD = 0.8

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
    parser.add_argument("-d", "--device", type=str, default='CPU',
                        help="Device to run inference on")
    return parser

def detect_face(frame, model):
    if frame is None:
        return None
    input_height, input_width, _ = frame.shape
    processed_frame=model.preprocess_input(frame)
    request_handle=model.predict(processed_frame,0)
    request_handle.wait()
    output=model.get_output(request_handle)
    boxes=model.preprocess_output(output,FACE_DETECTION_THRESHOLD,input_width,input_height)
    return utils.crop_image(frame,boxes)


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

    if not single_image_mode:
        while True:
            frame = next(input_feed.next_batch())
            
            cropped_faces = detect_face(frame, face_model)
            
            if cropped_faces==0: #no face detected, nothing to process
                continue
            elif cropped_faces is None: #finished reading input feed
                break
            elif len(cropped_faces)==1:
                cv2.imshow('face',cropped_faces[0])
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