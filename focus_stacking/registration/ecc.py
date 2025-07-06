import cv2, logging, tqdm
import numpy as np
import focus_stacking.utils.images as im_utils

logger = logging.getLogger(__name__)

def get_homography(im1, im2, max_iters, eps, init_warp=None):
    init_warp_matrix = init_warp if init_warp else np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, max_iters, eps)
    _, warp_matrix = cv2.findTransformECC(im1, im2, init_warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria)
    return warp_matrix

def do_perspective_warp(im, shape, warp_matrix):
    warped_im = cv2.warpPerspective(im, warp_matrix, shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return warped_im

'''
Align images to reference image
Downsize images by a factor of warp_res for purposes of computing warp
'''
def align(gray_images, images, ref_im_idx, warp_res, max_iters, eps, save_img_dir=None):
    logging.info(f'Aligning images to image {ref_im_idx}')
    gray_registered_images = []
    registered_images = []
    im_shape = images[ref_im_idx].shape[::-1]
    gray_im_shape = gray_images[ref_im_idx].shape[::-1]
    print(im_shape, gray_im_shape)

    # TODO: add option to downsize images for faster warping

    for im_idx, im in enumerate(tqdm.tqdm(images, desc='Performing image registration')):

        if im_idx == ref_im_idx:
            gray_registered_images.append(gray_images[im_idx])
            registered_images.append(images[im_idx])
            if save_img_dir:
                reg_im_path = f'{save_img_dir}/reg_{im_idx}.jpg'
                cv2.imwrite(reg_im_path, registered_images[im_idx])
        else:
            warp_matrix = get_homography(gray_images[ref_im_idx], gray_images[im_idx], max_iters, eps)
            gray_registered_images.append(do_perspective_warp(gray_images[im_idx], gray_im_shape, warp_matrix))
            registered_images.append(do_perspective_warp(images[im_idx], gray_im_shape, warp_matrix))
            if save_img_dir:
                reg_im_path = f'{save_img_dir}/reg_{im_idx}.jpg'
                cv2.imwrite(reg_im_path, registered_images[im_idx])


    return gray_registered_images, registered_images


if __name__ == "__main__":
    print("run")
