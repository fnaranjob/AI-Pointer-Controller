import cv2
from input_feeder import InputFeeder 

in_type='video'
path='../bin/demod.avi'

input_feed = InputFeeder(input_type=in_type,input_file=path)
frame=input_feed.load_data()

if not input_feed.cap.isOpened():
	print("error opening file")
	exit()

if len(frame)>0:
	cv2.imshow('frame',frame)
	cv2.waitKey(0)
else:
	while True:
		batch = input_feed.next_batch()
		cv2.imshow('frame',next(batch))
		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):
			break

input_feed.close()