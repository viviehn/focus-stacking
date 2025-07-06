import cv2, pywt
import numpy as np
import focus_stacking.utils.images as im_utils
import focus_stacking.energy.energy as energy

# https://stackoverflow.com/questions/60593425/how-to-view-the-pyramid-of-images-generated-with-opencv
def gen_pyramid_im(levels, out_path=None, process_im=lambda x: x, gray=False):
    h,w,c = levels[0].shape
    pyr_im = np.zeros((h, w+w//2, c), dtype=np.float32)
    '''
    level_im = process_im(levels[0])
    if gray:
        level_im = im_utils.bgr_to_grayscale(level_im.astype(np.float32))
        level_im = im_utils.grayscale_to_bgr(level_im)
        print('0 ', level_im)
    pyr_im[0:h, 0:w, :c] = level_im
    '''

    y_offset = 0
    x_offset = 0
    for i in range(0, len(levels)):
        if i == 1:
            x_offset = w
            y_offset = 0
        h_, w_, c_ = levels[i].shape
        level_im = process_im(levels[i])
        if gray:
            level_im = im_utils.bgr_to_grayscale(level_im.astype(np.float32))
            level_im = im_utils.grayscale_to_bgr(level_im)
            # print(f'{i} ', level_im)
        pyr_im[y_offset:y_offset+h_, x_offset:x_offset+w_, :c_] = level_im
        y_offset = y_offset + h_

    if out_path:
        cv2.imwrite(out_path, pyr_im)

# A stack of pyramids (where each index in the stack represents
# an image/slice from the focus stack)
class StackPyramid(object):
    def __init__(self, images, levels, kernel, pyramid_type='gaussian', wvt_name=None):
        if pyramid_type == 'gaussian':
            self.pyramids = [Pyramid(im, levels, kernel) for im in images]
        elif pyramid_type == 'laplacian':
            self.pyramids = []
            for im_id, im in enumerate(images):
                lap_pyr = LaplacianPyramid(im=im, num_levels=levels, kernel=kernel)
                lap_pyr.display_pyramid(f'/n/fs/3d-indoor/tmp_outdir/aux/pyr_gauss_{im_id}.jpg',
                                        f'/n/fs/3d-indoor/tmp_outdir/aux/pyr_lap_{im_id}.jpg')
                self.pyramids.append(lap_pyr)
        elif pyramid_type == 'wavelet':
            self.pyramids = []
            for im_id, im in enumerate(images):
                wvt_pyr = WaveletPyramid(im=im, num_levels=levels, wvt_name=wvt_name)
                self.pyramids.append(wvt_pyr)
        self.max_im_h = self.pyramids[0].max_im_h
        self.max_im_w = self.pyramids[0].max_im_h
        self.min_im_h = self.pyramids[0].min_im_h
        self.min_im_w = self.pyramids[0].min_im_h

class Pyramid(object):

    def __init__(self, im, num_levels, kernel=None, downsample=lambda im : im[::2, ::2, :]):
        self.num_levels = num_levels

        self.src_im = im
        self.max_im_h = self.src_im.shape[0]
        self.max_im_w = self.src_im.shape[1]
        self.min_im_h = self.max_im_h / (2**self.num_levels)
        self.min_im_w = self.max_im_w / (2**self.num_levels)

        self.kernel = kernel
        self.downsample = downsample
        self.levels = [{'image': self.src_im,
                        }]

        self.make_pyramid()

    def make_pyramid(self):
        for level_id in range(1, self.num_levels):
            prev_level = self.levels[level_id-1]['image']
            level = prev_level
            level = cv2.sepFilter2D(prev_level, -1, self.kernel, self.kernel)
            level = self.downsample(level)
            self.levels.append({'image': level})
            # TODO: optionally write out pyramid to image

    def display_pyramid(self, out_path):
        gen_pyramid_im(self.levels, out_path)

    def compute_energy_for_levels(self, level_id, im_key, energy_type, energy_key=None, ksize=5, kernel=None, im_color=True):
        if energy_key is None:
            energy_key = energy_type
        if energy_type == 'entropy':
            self.levels[level_id][energy_key] = energy.local_entropy(self.levels[level_id][im_key], ksize, im_color=im_color)
        elif energy_type == 'variance':
            self.levels[level_id][energy_key] = energy.local_variance(self.levels[level_id][im_key], ksize, im_color=im_color)
        elif energy_type == 'deviation':
            self.levels[level_id][energy_key] = energy.local_deviation(self.levels[level_id][im_key], ksize, im_color=im_color)
        elif energy_type == 'region':
            self.levels[level_id][energy_key] = energy.local_region(self.levels[level_id][im_key], kernel, im_color=im_color)
        else:
            print('unknown energy type specified')




class LaplacianPyramid(Pyramid):
    def __init__(self, im=None, num_levels=None, levels=None, kernel=None, downsample=lambda im: im[::2, ::2, :], upsample=energy.expand_from_kernel):
        self.upsample = upsample
        if levels is None:
            super().__init__(im, num_levels, kernel)
        else:
            self.num_levels = len(levels)
            self.levels = levels
            self.kernel = kernel
            # self.max_im_h = levels[0].shape[0]
            # self.max_im_w = levels[0].shape[1]
            # self.min_im_h = levels[-1].shape[0]
            # self.min_im_w = levels[-1].shape[1]

    def make_pyramid(self):
        print('making pyramid')
        super().make_pyramid()

        for level_id in range(self.num_levels - 1):
            upsampled = self.upsample(self.levels[level_id + 1]['image'], self.kernel)
            #self.levels.append(upsampled)
            diff = self.levels[level_id]['image'] - upsampled
            self.levels[level_id]['laplacian'] = diff
        self.levels[-1]['laplacian'] = self.levels[-1]['image']

    def display_pyramid(self, low_pass_out_path, out_path):
        gen_pyramid_im([self.levels[level_id]['image'] for level_id in range(self.num_levels)], low_pass_out_path)
        gen_pyramid_im([self.levels[level_id]['laplacian'] for level_id in range(self.num_levels)], out_path, 
                       process_im=im_utils.normalize_with_neg, gray=True)

    # Reconstruct an image based on self.levels
    def reconstruct(self):
        image = self.levels[-1]['laplacian']
        for level in self.levels[-2::-1]:
            image = self.upsample(image, self.kernel) + level['laplacian']
        return image


class WaveletPyramid(Pyramid):
    def __init__(self, im=None, num_levels=5, levels=None, wvt_name='db6'):
        self.wvt_name = wvt_name
        self.downsample = lambda im : pywt.dwt2(im, self.wvt_name)
        self.upsample = lambda coeff : pywt.idwt2(coeff, self.wvt_name)
        self.upsample_full = lambda coeff : pywt.waverec2(coeff, self.wvt_name)
        if levels is None:
            super().__init__(im, num_levels, kernel=None, downsample=self.downsample)
        else:
            self.num_levels = len(levels)
            self.levels = levels
            self.kernel = None
            
    def make_pyramid(self):
        print('making pyramid')
        LL = self.levels[0]['image']
        for level_id in range(self.num_levels):
            coeffs2 = self.downsample(self.levels[level_id]['image'])
            LL, (LH, HL, HH) = coeffs2
            self.levels[level_id]['LH'] = LH
            self.levels[level_id]['HL'] = HL
            self.levels[level_id]['HH'] = HH
            self.levels.append({'image': LL})
            
    def reconstruct(self):
        coeffs = [self.levels[-1]['image']]+[(l['LH'], l['HL'], l['HH']) for l in self.levels[-2::-1]]
        for l in self.levels[-2::-1]:
            print(l['LH'].shape, l['HL'].shape, l['HH'].shape)
        image = self.upsample_full(coeffs)
        return image