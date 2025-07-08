import numpy as np
from skimage.measure import shannon_entropy as shannon_entropy
from skimage.filters.rank import entropy as entropy
from skimage.morphology import disk
import cv2

#low_pass_kernel = cv2.getGaussianKernel(ksize, sigma=-1)

def expand_from_kernel(im, kernel):
    new_im = np.zeros((im.shape[0]*2, im.shape[1]*2, im.shape[2]))
    new_im[::2, ::2, :] = im
    new_im = cv2.sepFilter2D(new_im, -1, kernel, kernel)
    return 4*new_im

#https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/#h-focus-operators-and-their-computation
#??????

def local_variance(im, neighborhood, im_color=True):
    if im_color:
        gray_im = cv2.cvtColor(im.astype(np.float32), cv2.COLOR_BGR2GRAY) / 255
    else:
        gray_im = im
    # cv2.blur is a box filter so serves to average over the window given by ksize
    mean = cv2.blur(gray_im, (neighborhood, neighborhood))
    squared_mean = cv2.blur(gray_im**2, (neighborhood, neighborhood))
    variance = squared_mean - (mean**2)
    return variance

# possible alt: computation done in blocks (i.e. one average value for the whole block)
# based on https://scispace.com/pdf/a-multi-focus-image-fusion-method-based-on-laplacian-pyramid-2rnkrqtoeq.pdf
# https://github.com/sjawhar/focus-stacking/blob/master/focus_stack/pyramid.py
def local_deviation(image, neighborhood):
    def _area_deviation(area):
        average = np.average(area).astype(np.float64)
        return np.square(area - average).sum() / area.size

    pad_amount = int((neighborhood - 1) / 2)
    padded_image = cv2.copyMakeBorder(image,pad_amount,pad_amount,pad_amount,pad_amount,cv2.BORDER_REFLECT101)
    deviations = np.zeros(image.shape[:2], dtype=np.float64)
    offset = np.arange(-pad_amount, pad_amount + 1)
    for row in range(deviations.shape[0]):
        for column in range(deviations.shape[1]):
            area = padded_image[row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset]
            deviations[row, column] = _area_deviation(area)

    return deviations

def gray_level_probabilities(gray_im):
    gray_levels, counts = np.unique(gray_im.astype(np.uint8), return_counts=True)
    probabilities = np.empty((256,), dtype=np.float64)
    probabilities[gray_levels] = counts.astype(np.float64)/counts.sum()
    return probabilities


# TODO: debug?
def local_entropy(im, neighborhood, im_color=True):
    if im_color:
        gray_im = cv2.cvtColor(im.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray_im = im

    probabilities = gray_level_probabilities(gray_im)
    information = np.log(probabilities[gray_im.astype(np.uint8)])
    # entropy = -1.*probabilities[gray_im.astype(np.uint8)]*cv2.boxFilter(information.astype(np.float64), -1, (ksize,ksize))
    # entropy = -1.*cv2.boxFilter((gray_im.astype(np.float64) * information).astype(np.float64), -1, (ksize,ksize))
    # entr_img = entropy
    entr_img = entropy(gray_im.astype(np.float32)/255, disk(neighborhood))
    return entr_img

def local_global_entropy(im, neighborhood, im_color=True):
    if im_color:
        gray_im = cv2.cvtColor(im.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray_im = im.astype(np.uint8)

    probabilities = gray_level_probabilities(gray_im)
    information = np.log(probabilities[gray_im.astype(np.uint8)])
    # entropy = -1.*probabilities[gray_im.astype(np.uint8)]*cv2.boxFilter(information.astype(np.float64), -1, (ksize,ksize))
    # entropy = -1.*cv2.boxFilter((gray_im.astype(np.float64) * information).astype(np.float64), -1, (ksize,ksize))
    # entr_img = entropy
    entr_img = entropy(gray_im.astype(np.float32)/255, disk(neighborhood))
    return entr_img

# possible alt: computation done in blocks (i.e. one average value for the whole block)
# apply filter given by kernel then take average over neighborhood
def local_region(im, kernel, neighborhood=None, im_color=True):
    if im_color:
        gray_im = cv2.cvtColor(im.astype(np.float32), cv2.COLOR_BGR2GRAY)
    else:
        gray_im = im
    region_energy = cv2.sepFilter2D(gray_im**2, -1, kernel, kernel)
    if neighborhood is None:
        neighborhood = len(kernel)
    region_energy = cv2.blur(region_energy, (neighborhood, neighborhood))
    return region_energy

def laplacian(im, ksize):
    return

def squared(im, im_color=True):
    return im**2
