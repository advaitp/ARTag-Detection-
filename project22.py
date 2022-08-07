from project21 import detect, find_ar, transform, decode_ar
import cv2
from scipy import fftpack
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, fft2, ifft2,ifftshift
import matplotlib.pyplot as plt
from Utils import *

def stitchImage(img1, img2):
	img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	img1c=img1.copy()

	img1[:,:,0] = mask_inv
	img1[:,:,1] = mask_inv
	img1[:,:,2] = mask_inv
	
	final_mask = cv2.bitwise_and(img1,img1c)
	dst = cv2.add(img2, final_mask)
	return dst


def project(H, K, cube_pts) :
	B_ = (np.linalg.inv(K))@H
	if np.linalg.det(B_) < 0 : -1*B_

	b1, b2, b3 = B_[:, 0], B_[:, 1], B_[:, 2]
	lambd = 2/(np.linalg.norm(b1)+np.linalg.norm(b2))

	B = lambd*B_

	b1n, b2n, b3n = B[:, 0], B[:, 1], B[:, 2]
	r1, r2, t = lambd*b1n, lambd*b2n, lambd*b3n
	r3 = np.cross(r1,r2)

	RT = np.array([r1, r2, r3, t]).T

	P = K@RT
	ppts = P@cube_pts.T

	return ppts 

def superimpose(img, testudoimg) :
	try : 
		im = img.copy()
		im1 = img.copy()

		# get the fft of the image
		i_img = detect(im)  

		# cpts are in counterclockwise direction
		cpts = find_ar(i_img, im)

		# Decoding the April tag
		height, width = 200, 200
		tr_img = transform(img, cpts, height, width)
		tr_img = cv2.resize(tr_img, (0,0), fx=0.04, fy=0.04)
		
		# plt.imshow(tr_img, cmap='gray')
		# plt.show()

		# Getting the Id and orientat5ion of tag to rotate testudo image
		ar_id, ar_img, orient = decode_ar(tr_img)
		print(f'Id of April Tag {ar_id} and Orientation is {orient}')

		# Rotating the testudo image according to orientation
		num = orient // 90 
		for i in range(num) :
			testudoimg = cv2.rotate(testudoimg, cv2.ROTATE_90_COUNTERCLOCKWISE)

		# Gettig corner points of testudo image
		height, width = testudoimg.shape[0], testudoimg.shape[1]
		testudopts = np.array([[0,0], [0,height],[width,height],[width,0]]).reshape((-1,1,2))

		# Finding homography
		# H, mask = cv2.findHomography(testudopts, cpts)
		H = findHomography(testudopts, cpts)
		
		# Warping the testudo image
		imheight, imwidth = im.shape[0], im.shape[1]
		# w_timg = cv2.warpPerspective(testudoimg, H, (imwidth, imheight), flags=cv2.INTER_LINEAR)
		w_timg = warpPerspective(testudoimg, H, (imwidth, imheight))

		# Superimposing the warped and original base image
		st_img = stitchImage(img, w_timg)

		# Projection matrix
		# K is camera intrinsic matrix
		cube_pts = np.array([[0,0,0,1], [0,height,0,1],[width,height,0,1],[width,0,0,1],
			[0,0,-0.2,1], [0,height,-0.2,1],[width,height,-0.2,1],[width,0,-0.2,1]])

		K = np.array([[1346.1005,0,932.163],[0,1355.933,654.89867],[0,0,1]])
		ppts = project(H, K, cube_pts)

		x1,y1,z1 = ppts[:,0]
		x2,y2,z2 = ppts[:,1]
		x3,y3,z3 = ppts[:,2]
		x4,y4,z4 = ppts[:,3]
		x5,y5,z5 = ppts[:,4]
		x6,y6,z6 = ppts[:,5]
		x7,y7,z7 = ppts[:,6]
		x8,y8,z8 = ppts[:,7]

		cv2.line(st_img,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,0,0), 2)
		cv2.line(st_img,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
		cv2.line(st_img,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
		cv2.line(st_img,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)

		cv2.line(st_img,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
		cv2.line(st_img,(int(x1/z1),int(y1/z1)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
		cv2.line(st_img,(int(x2/z2),int(y2/z2)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
		cv2.line(st_img,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

		cv2.line(st_img,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
		cv2.line(st_img,(int(x5/z5),int(y5/z5)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
		cv2.line(st_img,(int(x6/z6),int(y6/z6)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
		cv2.line(st_img,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
		
		return st_img

	except Exception as e :
		print(f'Exception in superimpose function : {e}')
		return img

if __name__ == "__main__" :
	img = cv2.imread('f50.jpg')
	testudoimg = cv2.imread('testudo.png')
	superimposed = superimpose(img, testudoimg)
