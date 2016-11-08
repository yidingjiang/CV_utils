import cv2
import numpy as np
import math

def adaptiveNormalization(image):
	"""
	Apply adaptive normalization on all channels of image and merge
	the channels

	Input:
	image -- a numpy array representation of the image

	Output:
	image processed by adaptiveNormalization
	"""
	b, g, r = cv2.split(img)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl_b = clahe.apply(b)
	cl_g = clahe.apply(g)
	cl_r = clahe.apply(r)
	cl_img = cv2.merge((cl_b, cl_g, cl_r))
	return cl_img

def lecunLCN(image, ksize=9, sigma=3):
	"""
	Implementation of local contrast normaliztion by Yann LeCun

	Input:
	image -- a numpy array representation of the image with 3 color channels
	ksize -- s of the gaussian kernel
	sigma -- the standard deviation of the gaussian kernel

	Output:
	Image processed by local contrast normaliztion
	"""
	kernel = cv2.getGaussianKernel(ksize, sigma)
	gaussian_avg = cv2.filter2D(image, -1, kernel).sum(axis=(2))/3.0
	diff = image - np.dstack((gaussian_avg, gaussian_avg, gaussian_avg))
	sq_diff = diff**2
	std = np.sqrt((cv2.filter2D(sq_diff, -1, kernel).sum(axis=(2))))
	c = cv2.blur(std, (9, 9))/(81.0)
	stack_std, stack_c = np.dstack((std,std,std)), np.dstack((c,c,c))
	output = diff/np.maximum(stack_std, stack_c, 0.0001*np.ones(stack_std.shape))
	return output

