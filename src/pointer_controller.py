import numpy as np
import logging as log
from argparse import ArgumentParser

from input_feeder import InputFeeder 
import face_detection
import facial_landmarks_detection
import gaze_estimation
import head_pose_estimation

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--input_type", required=True, type=str,
                        help="Input type, use 'cam' for camera, 'image' for single image, 'video' for video file")
    parser.add_argument("-p", "--input_path", type=str, default=None, 
    					help="Path to input, unused if input_type='cam'")
    return parser


def main():
    args = build_argparser().parse_args()
    
    input_feed = InputFeeder(input_type=args.input_type,input_file=args.input_path)
    input_feed.load_data()

    if not input_feed.is_open():
        log.critical('Error opening input, check --input_path parameter (use --help for more info)')
        exit()

if __name__ == '__main__':
    main()