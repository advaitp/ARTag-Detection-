import cv2
from scipy import fftpack
import numpy as np
from scipy.fft import fft, fftfreq, fftshift, fft2, ifft2,ifftshift
import matplotlib.pyplot as plt

def findHomography(p1,p2):
	A = []
	p1 = p1.reshape((-1,2))
	p2 = p2.reshape((-1,2))
	for i in range(0, len(p1)):
		x, y = p1[i][0], p1[i][1]
		u, v = p2[i][0], p2[i][1]
		A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
		A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

	A = np.asarray(A)
	U, S, Vh = np.linalg.svd(A) 
	L = Vh[-1,:] / Vh[-1,-1]
	H = L.reshape(3, 3)

	return H

def warpPerspective(image, H, size):
	try : 
		icols, irows = size
		h, w = image.shape[0], image.shape[1]
		yp, xp = np.indices((h, w))
		xi, yi = xp.ravel(), yp.ravel()
		img_pts = np.stack([xi, yi, np.ones((xp.size))])

		warp_pts = H@img_pts
		warp_pts = warp_pts/(warp_pts[2,:]+1e-6)
		warp_pts = np.round(warp_pts).astype(int)

		xc = warp_pts[0,:]
		yc = warp_pts[1,:]

		xc[xc<0] = 0 
		yc[yc<0] = 0

		xc[xc>=icols] = icols-1
		yc[yc>=irows] = irows-1

		new_img = np.zeros((irows, icols, 3))
		new_img[yc, xc] = image[yi, xi]

		new_img = new_img.astype('uint8')

		return new_img
	except Exception as e: 
		print(f'Exception in warpPerspective : {e}')
		return image