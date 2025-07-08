import numpy as np
import cv2


def sobel_mag(im, ksize):
    im_sobel_x = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize).max(axis=-1)
    im_sobel_y = cv2.Sobel(im, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize).max(axis=-1)
    return np.sqrt(im_sobel_x**2 + im_sobel_y**2), (im_sobel_x, im_sobel_y)

def relative_gradients(im1, im2, ksize):
    im1_gradient_mag, _, = sobel_mag(im1, ksize)
    im2_gradient_mag, _ = sobel_mag(im2, ksize)
    rel_mag = np.where(im1_gradient_mag < im2_gradient_mag, 
                       im1_gradient_mag/(im2_gradient_mag+1e-9),
                       im2_gradient_mag/(im1_gradient_mag+1e-9))
    return rel_mag

def gradient_preservation(im1, im2, ksize, gamma=1, k=-10, sigma=0.5):
    rel_gradient_mag = relative_gradients(im1, im2, ksize)
    gradient_preservation = gamma / (1 + np.e**(k*(rel_gradient_mag-sigma)))
    score = np.mean(gradient_preservation)
    return rel_gradient_mag, gradient_preservation, score