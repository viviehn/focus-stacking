import numpy as np
import os, glob, io
import argparse, cv2, logging, pdb
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import color


def compute_ssim(im1, im2):
    ssim_avg, ssim_grad, ssim_full = ssim(im1, im2,
                                          data_range=1.,
                                          gradient=True, full=True,
                                          )
    return ssim_avg, ssim_full
    #st.pyplot(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--im1', type=str, help='full path to im1')
    parser.add_argument('--im2', type=str, help='full path to im2')
    parser.add_argument('--out_dir', type=str, help='path to output eval dir')
    parser.add_argument('--format_fig', action='store_true', help='save a formated figure (with score printed on it)')
    parser.add_argument('--ssim', action='store_true', help='compute SSIM between im1 and im2')
    parser.add_argument('--mse', action='store_true', help='compute MSE between im1 and im2')
    parser.add_argument('--rmse', action='store_true', help='compute RMSE between im1 and im2')
    parser.add_argument('--psnr', action='store_true', help='compute PSNR between im1 and im2')

    im1_basename = os.path.basename(args.im1)
    im2_basename = os.path.basename(args.im2)
    out_path = f'{args.out_dir}/{im1_basename}_{im2_basename}'

    im1 = cv2.imread(args.im1)
    im2 = cv2.imread(args.im2)

    metric_report = '''
    ======================
    RESULTS
    ======================
    '''

    if args.ssim:
        ssim_out_path = f'{out_path}_ssim.png'
        ssim_score, ssim_im = compute_ssim(im1, im2)
        cv2.imwrite(ssim_im, ssim_out_path)
        if args.format_fig:
            # TODO: format a matplotlib plot
            fig = plt.figure()
            plt.axis('off')
            plt.tight_layout()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.imshow(ssim_full, vmin=-1, vmax=1, cmap='jet')
            plt.title(f'SSIM={ssim_score:.4f}')
        metric_report += f'''
        SSIM: {ssim_score:.4f}
        '''

    print(metric_report)

