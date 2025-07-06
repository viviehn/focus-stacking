import cv2, glob, os
import logging
logger = logging.getLogger(__name__)

def get_src_dir(args, DATA_ROOT):
    if args.obj and args.view and args.fnum:
        src_dir = f'{DATA_ROOT}/{args.obj}/{args.view}/{args.fnum}/images'
        if os.path.exists(src_dir):
            return src_dir
    if not dir_exists and args.src_dir:
        src_dir = args.src_dir
        if os.path.exists(src_dir):
            return src_dir
    if not dir_exists and args.src_dir:
        src_dir = f'{DATA_ROOT}/{args.src_dir}'
        if os.path.exists(src_dir):
            return src_dir
    return None

def load_images_from_dir(src_dir, match='*.JPG'):
    logging.info(f'Loading images from {src_dir}')
    im_names = glob.glob(f'{src_dir}/{match}')
    logging.info(f'Loading images: {im_names}')
    im_names.sort()
    images = [cv2.imread(im_name) for im_name in im_names]
    return images

