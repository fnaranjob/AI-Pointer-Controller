import numpy as np 

top_left = np.array([ 0.5, 0.01])
top_right = np.array([-0.5,  0.01])
bottom_left = np.array([ 0.6, -0.4])
bottom_right = np.array([-0.5, -0.4])

np.savez('calibration.npz', top_left=top_left, top_right=top_right, bottom_left=bottom_left, bottom_right=bottom_right)