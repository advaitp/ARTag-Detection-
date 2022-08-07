from project21 import *
import cv2
from scipy import fftpack
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, fft2, ifft2,ifftshift
import matplotlib.pyplot as plt
from project22 import *
from Utils import *

cap = cv2.VideoCapture('1tagvideo.mp4')
frames = []
if (cap.isOpened()== False): 
	print("Error opening video stream or file")
   
testudoimg = cv2.imread('testudo.png')
framenum = 0
video= cv2.VideoWriter('ARTag.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1920,1080))
i = 0
try : 
	while(cap.isOpened() or i < 785):
		ret, frame = cap.read()
		super_frame = superimpose(frame, testudoimg)
		video.write(super_frame)
		cv2.imshow('Super', super_frame)
		cv2.waitKey(1)
		i += 1
		if ret == True:
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
			framenum += 1
			print(f'Frame Number : {framenum}')
		else: 
			break
except : 
	print('Exception')

cap.release()
cv2.destroyAllWindows()
video.release()