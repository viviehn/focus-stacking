import argparse, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

def gen_ssim_plot(im1_path, im2_path, output_path):
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)

    im1_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(im1_bw, im2_bw, full=True)
    print('SSIM:', score)

    diff = np.array(diff)

    print(diff.min(), np.quantile(diff, .1), np.quantile(diff, .9), diff.max())
    fig = plt.figure()
    plt.axis('off')
    plt.title(f'SSIM: {score}')
    im_ratio = diff.shape[0] / diff.shape[1]
    im = plt.imshow(diff, cmap='Spectral', vmin=-1, vmax=1)
    fig.colorbar(im, fraction=0.047*im_ratio)
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--run_single', action='store_true')
    parser.add_argument('--im1',type=str)
    parser.add_argument('--im2',type=str)
    parser.add_argument('--output_path',type=str)
    args = parser.parse_args()


    if args.run_single:
        gen_ssim_plot(args.im1, args.im2, args.output_path)

    else:

        subdirs = glob.glob(f'{args.dir}/*')
        for subdir in subdirs:
            gen_ssim_plot(f'{subdir}/narrow_stacked.jpg',
                          f'{subdir}/wide_stacked.jpg',
                          f'{subdir}/ssim_diff.png')

