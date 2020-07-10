import numpy as np
import pyautogui
import cv2

COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255),(128,0,0),(0,128,0),(0,0,128)]
LINE_THICKNESS = -1 #filled
FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5

SCREEN_WIDTH = pyautogui.size().width
SCREEN_HEIGHT = pyautogui.size().height
BOX_SIDE_LENGTH = 80 

top_left_square = {'pt1':(0,0), 'pt2':(BOX_SIDE_LENGTH,BOX_SIDE_LENGTH)}
top_right_square = {'pt1':(SCREEN_WIDTH - BOX_SIDE_LENGTH,0), 'pt2':(SCREEN_WIDTH, BOX_SIDE_LENGTH)}
bottom_left_square = {'pt1':(0,SCREEN_HEIGHT - BOX_SIDE_LENGTH), 'pt2':(BOX_SIDE_LENGTH,SCREEN_HEIGHT)}
bottom_right_square = 	{'pt1':(SCREEN_WIDTH - BOX_SIDE_LENGTH,SCREEN_HEIGHT - BOX_SIDE_LENGTH), 
						'pt2':(SCREEN_WIDTH,SCREEN_HEIGHT)}

cal_squares = [top_left_square,top_right_square,bottom_left_square, bottom_right_square]
def imshow_fullscreen (winname, img):
	cv2.namedWindow (winname, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty (winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.imshow (winname, img)
	cv2.waitKey(0)


base_img = np.zeros([SCREEN_HEIGHT,SCREEN_WIDTH,3],dtype=np.uint8)
base_img.fill(255)
cv2.putText(base_img, "LOOK AT THE SQUARES FOR 3 SECONDS", (SCREEN_WIDTH//2-450,SCREEN_HEIGHT//2-20), FONT, FONT_SCALE, COLORS[0], FONT_THICKNESS)
cv2.putText(base_img, "AND THEN PRESS ANY KEY", (SCREEN_WIDTH//2-300,SCREEN_HEIGHT//2+20), FONT, FONT_SCALE, COLORS[0], FONT_THICKNESS)

for square in cal_squares:
	img = np.copy(base_img)
	cv2.rectangle(img,square['pt1'], square['pt2'],COLORS[0],-1)
	imshow_fullscreen('window',img)

cv2.destroyAllWindows()