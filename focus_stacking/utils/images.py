import cv2
import numpy as np

def bgr_to_grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def grayscale_to_bgr(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


# image with negative values 
def normalize_with_neg(im):
    range_im = (((im / 255) +1) / 2) * 255
    normalize_im = (range_im - range_im.min() / (range_im.max() - range_im.min()))
    return normalize_im

# to display images with matplotlib
def bgr_to_rgb(im):
    return im[...,::-1]

def gaussian_blur(im, ksize=5):
    kernel = cv2.getGaussianKernel(ksize, sigma=-1)
    blurred_im = cv2.sepFilter2D(im, -1, kernel, kernel)
    return blurred_im
    
def laplacian(im, ksize=5):
    blurred_im = gaussian_blur(im, ksize)
    diff = im - blurred_im
    return blurred_im, diff