import cv2
from scipy import fftpack, stats
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, fft2, ifft2,ifftshift
import matplotlib.pyplot as plt
from Utils import *

# creating mask of high pass filter
def createMask(img) :
	rows, cols = img.shape[0], img.shape[1]
	crow, ccol = int(rows / 2), int(cols / 2)
	r = 400
	r1 = 900
	mask = np.ones((rows, cols))
	center = [crow, ccol]
	x, y = np.ogrid[:rows, :cols]
	mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
	mask_area1 = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r1*r1
	mask[mask_area] = 0
	mask[mask_area1] = 1
	return mask

# edge detection 
def detect(img) :
	gr = sharp(img)

	# Creating Fourier transform
	gr_fourier = fftshift(fft2(gr))
	magnitude_spectrum = 20 * np.log(abs(gr_fourier))

	# Creating high pass mask
	hpmask = createMask(gr)

	# Applying mask
	fshift = gr_fourier * hpmask
	
	# Taking inverse fourier
	f_ishift = ifftshift(fshift)
	i_img = ifft2(f_ishift)
	i_img = (np.abs(i_img)).astype(np.uint8)
	return i_img

def find_ar(image, or_image) : 
	or_img = or_image.copy()
	imgc = or_img.copy()
	ret,img = cv2.threshold(image, 175 ,255,cv2.THRESH_BINARY)

	corners = cv2.goodFeaturesToTrack(img,10000,0.01,1)
	corners = np.int0(corners)

	for i in corners:
		x,y = i.ravel()
		# cv2.circle(or_img,(x,y),3,255,-1)
	corners = corners.reshape((-1,2))

	a1 = np.abs(stats.zscore(corners[:,0], axis = 0)) < 3.15
	a2 = np.abs(stats.zscore(corners[:,1], axis = 0)) < 3.15
	c = np.row_stack((a1,a2)).T

	corners = corners[c.all(axis=1)]
	minx = corners[np.argmin(corners[:,0], axis=0)]
	maxx = corners[np.argmax(corners[:,0], axis=0)]
	miny = corners[np.argmin(corners[:,1], axis=0)]
	maxy = corners[np.argmax(corners[:,1], axis=0)]

	rect = or_img[miny[1]:maxy[1], minx[0]:maxx[0]]
	xmin,xmax,ymin,ymax = 1920, 0, 1920, 0
	xmin2,xmax2,ymin2,ymax2 = 1920, 0, 1920, 0

	ci = np.array([miny, minx, maxy, maxx])
	or_img1 = or_img.copy()
	imgc = cv2.polylines(or_img, [ci], True, (255, 0, 0), 75)
	imgrc = cv2.fillPoly(or_img1, [ci], (0, 0, 255))

	i=0
	for c in corners:
		x,y = c
		if imgrc[y,x,0] != 0 and imgrc[y,x,1] != 0 and imgrc[y,x,2] != 255 :
			corners = np.delete(corners,i,0)
			i=i-1

		elif imgc[y,x,0] == 255 and  imgc[y,x,1] == 0 and  imgc[y,x,2] == 0:
			corners = np.delete(corners,i,0)
			i=i-1
		i=i+1

	cornerf = corners
	rect = np.zeros((4, 2), dtype = "float32")
	s = cornerf.sum(axis = 1)
	rect[1] = cornerf[np.argmin(s)]
	rect[3] = cornerf[np.argmax(s)]
	diff = np.diff(cornerf, axis = 1)
	rect[0] = cornerf[np.argmin(diff)]
	rect[2] = cornerf[np.argmax(diff)]

	return rect

def transform(img, pts, height, width) :
	try : 
		world_pts = np.array([[0,0], [0,height],[width,height],[width,0]]).reshape((-1,1,2))
		pts = pts.reshape((-1,1,2))

		pts = pts.reshape((-1,2))
		world_pts = np.array([[0,0], [0,height],[width,height],[width,0]]).reshape((-1,2))
		H = findHomography(pts, world_pts)
		tr_img = warpPerspective(img, H, (width, height))
		return tr_img
	except Exception as e :
		print(f'Exception in transform function : {e}')
		return img

def decode_ar(img) :
	img = cv2.resize(img, (200,200), cv2.INTER_AREA)
	im = img.copy()
	ret,im = cv2.threshold(im, 127 ,255,cv2.THRESH_BINARY)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	tag_id = 0
	orient = 0

	# Left Top Corner
	if int(im[50:75,50:75].squeeze().sum()/625) > 127:
		print('Left Top Corner')
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		orient = 180

	# Right Top Corner
	elif int(im[50:75,125:150].squeeze().sum()/625) > 127:
		print('Right Top Corner')
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		orient = 90

	# Left Bottom Corner
	elif int(im[125:150,50:75].squeeze().sum()/625) > 127:
		print('Left Bottom Corner')
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		orient = 270

	patch1 = img[75:100,75:100]
	patch2 = img[75:100,100:125]
	patch3 = img[100:125,100:125]
	patch4 = img[100:125,75:100]

	# print(int(patch1.sum()/625))
	# print(int(patch2.sum()/625))
	# print(int(patch3.sum()/625))
	# print(int(patch4.sum()/625))
	if int(patch1.squeeze().sum()/625) > 450 : tag_id += pow(2,0)
	if int(patch2.squeeze().sum()/625) > 450 : tag_id += pow(2,1)
	if int(patch3.squeeze().sum()/625) > 450: tag_id += pow(2,2)
	if int(patch4.squeeze().sum()/625) > 450 : tag_id += pow(2,3)
	
	return tag_id, img, orient

def sharp(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,image_sharp = cv2.threshold(img,190,255,cv2.THRESH_BINARY)
	kernel = np.array([[-1, -1, -1],
	  [-1, 10,-1],
	  [-1, -1, -1]])
	image_sharp = cv2.filter2D(image_sharp, ddepth=-1, kernel=kernel)
	return image_sharp


if __name__ == "__main__" :
	img = cv2.imread('f414.jpg')

	i_img = detect(img)  
	
	pts = find_ar(i_img, img)

	height, width = 200, 200
	tr_img = transform(img, pts, height, width)
	
	ar_id, ar_img, orient = decode_ar(tr_img)
	print(f'Id of AR Tag : {ar_id}')