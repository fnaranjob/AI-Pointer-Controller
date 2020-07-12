'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
import logging as log
from numpy import ndarray

class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
    
    def load_data(self):
        if self.input_type=='video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap=cv2.VideoCapture(0)
        else:
            self.frame = cv2.imread(self.input_file)
            self.cap=None

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        Not to be used if input_type is 'image', in that case the 
        image can be accessed directly through the .frame attribute
        '''
        if not self.input_type == 'image':
            while True:
                for _ in range(1):
                    _, frame=self.cap.read()
                yield frame
        else:
            log.critical("Illegal call, method only supported for 'video' and 'cam' input types")
            sys.exit(1)

    def is_open(self):
        if self.input_type == 'video' or self.input_type == 'cam':
            if not self.cap.isOpened():
                return False
        else:
            if self.frame is None:
                return False
        return True

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

