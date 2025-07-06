import cv2, pywt
import numpy as np
import focus_stacking.utils.images as im_utils
import focus_stacking.energy.energy as energy
import focus_stacking.fusion.fusion as fusion 
import focus_stacking.pipelines.utils as pipeline_utils

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
    def __init__(self, images, depth, kernel, pyramid_type='gaussian', wvt_name=None):
        print(pyramid_type)
        if pyramid_type == 'gaussian':
            self.pyramids = [Pyramid(im, depth, kernel) for im in images]
        elif pyramid_type == 'laplacian_pyramid':
            self.pyramids = []
            for im_id, im in enumerate(images):
                lap_pyr = LaplacianPyramid(im=im, depth=depth, kernel=kernel)
                lap_pyr.display_pyramid(f'/n/fs/3d-indoor/tmp_outdir/aux/pyr_gauss_{im_id}.jpg',
                                        f'/n/fs/3d-indoor/tmp_outdir/aux/pyr_lap_{im_id}.jpg')
                self.pyramids.append(lap_pyr)
        elif pyramid_type == 'wavelet':
            self.pyramids = []
            for im_id, im in enumerate(images):
                wvt_pyr = WaveletPyramid(im=im, depth=depth, wvt_name=wvt_name)
                self.pyramids.append(wvt_pyr)
        self.max_im_h = self.pyramids[0].max_im_h
        self.max_im_w = self.pyramids[0].max_im_h
        self.min_im_h = self.pyramids[0].min_im_h
        self.min_im_w = self.pyramids[0].min_im_h
        self.depth = depth

class Pyramid(object):

    def __init__(self, im, depth, kernel=None, downsample=lambda im : im[::2, ::2, :]):
        self.depth = depth

        self.src_im = im
        self.max_im_h = self.src_im.shape[0]
        self.max_im_w = self.src_im.shape[1]
        self.min_im_h = self.max_im_h / (2**self.depth)
        self.min_im_w = self.max_im_w / (2**self.depth)

        self.kernel = kernel
        self.downsample = downsample
        self.levels = [{'image': self.src_im,
                        }]

        self.make_pyramid()

    def make_pyramid(self):
        for level_id in range(1, self.depth):
            prev_level = self.levels[level_id-1]['image']
            level = prev_level
            level = cv2.sepFilter2D(prev_level, -1, self.kernel, self.kernel)
            level = self.downsample(level)
            self.levels.append({'image': level})
            # TODO: optionally write out pyramid to image

    def display_pyramid(self, out_path):
        gen_pyramid_im(self.levels, out_path)

    def compute_energy_for_levels(self, level_id, im_key, energy_fn, energy_key=None, ksize=5, kernel=None, im_color=True):
        if energy_key is None:
            energy_key = energy_type
        self.levels[level_id][energy_key] = energy_fn(self.levels[level_id][im_key], im_color)




class LaplacianPyramid(Pyramid):
    def __init__(self, im=None, depth=None, levels=None, kernel=None, downsample=lambda im: im[::2, ::2, :], upsample=energy.expand_from_kernel):
        self.upsample = upsample
        if levels is None:
            super().__init__(im, depth, kernel)
        else:
            self.depth = len(levels)
            self.levels = levels
            self.kernel = kernel
            # self.max_im_h = levels[0].shape[0]
            # self.max_im_w = levels[0].shape[1]
            # self.min_im_h = levels[-1].shape[0]
            # self.min_im_w = levels[-1].shape[1]

    def make_pyramid(self):
        print('making pyramid')
        super().make_pyramid()

        for level_id in range(self.depth - 1):
            upsampled = self.upsample(self.levels[level_id + 1]['image'], self.kernel)
            #self.levels.append(upsampled)
            diff = self.levels[level_id]['image'] - upsampled
            self.levels[level_id]['laplacian'] = diff
        self.levels[-1]['laplacian'] = self.levels[-1]['image']

    def display_pyramid(self, low_pass_out_path, out_path):
        gen_pyramid_im([self.levels[level_id]['image'] for level_id in range(self.depth)], low_pass_out_path)
        gen_pyramid_im([self.levels[level_id]['laplacian'] for level_id in range(self.depth)], out_path, 
                       process_im=im_utils.normalize_with_neg, gray=True)

    # Reconstruct an image based on self.levels
    def reconstruct(self):
        image = self.levels[-1]['laplacian']
        for level in self.levels[-2::-1]:
            image = self.upsample(image, self.kernel) + level['laplacian']
        return image


class WaveletPyramid(Pyramid):
    def __init__(self, im=None, depth=5, levels=None, wvt_name='db6'):
        self.wvt_name = wvt_name
        self.downsample = lambda im : pywt.dwt2(im, self.wvt_name)
        self.upsample = lambda coeff : pywt.idwt2(coeff, self.wvt_name)
        self.upsample_full = lambda coeff : pywt.waverec2(coeff, self.wvt_name)
        if levels is None:
            super().__init__(im, depth, kernel=None, downsample=self.downsample)
        else:
            self.depth = depth
            self.levels = levels
            self.kernel = None
            
    def make_pyramid(self):
        print('making pyramid')
        LL = self.levels[0]['image']
        for level_id in range(self.depth):
            coeffs2 = self.downsample(self.levels[level_id]['image'])
            LL, (LH, HL, HH) = coeffs2
            self.levels[level_id]['LH'] = LH
            self.levels[level_id]['HL'] = HL
            self.levels[level_id]['HH'] = HH
            self.levels.append({'image': LL})
            
    def reconstruct(self):
        coeffs = ([self.levels[-1]['image']] +
                [(l['LH'],
                  l['HL'],
                  l['HH']) for l in self.levels[-2::-1]])
        for l in self.levels[-2::-1]:
            print(l['LH'].shape, l['HL'].shape, l['HH'].shape)
        image = self.upsample_full(coeffs)
        return image

def run_laplacian_pyramid_pipeline(args, registered_images):
    low_pass_kernel = cv2.getGaussianKernel(args.st_kwidth, sigma=-1)
    stack = StackPyramid(registered_images,
            depth=args.st_depth,
            kernel=low_pass_kernel,
            pyramid_type=args.st_type
            )
    for energy_name in args.st_level_energies:
        fn = pipeline_utils.get_energy_fn(energy_name, args)
        for pyramid in stack.pyramids:
            for level in range(pyramid.depth):
                pyramid.compute_energy_for_levels(level,
                        im_key='laplacian',
                        energy_fn=fn,
                        energy_key=energy_name
                        )

    for energy_name in args.st_base_energies:
        fn = pipeline_utils.get_energy_fn(energy_name, args)
        for pyramid in stack.pyramids:
            pyramid.compute_energy_for_levels(-1,
                    im_key='laplacian',
                    energy_fn=fn,
                    energy_key=energy_name
                    )

    base_ims = [pyramid.levels[-1]['laplacian'] for pyramid in stack.pyramids]
    # TODO: generalize to any base energies, fusion strategy
    base_entropy = [pyramid.levels[-1]['local_entropy'] for pyramid in stack.pyramids]
    base_variance = [pyramid.levels[-1]['local_variance'] for pyramid in stack.pyramids]

    base_fusion_fn = pipeline_utils.get_fusion_fn(args.st_base_fusion, args)
    fused_base = base_fusion_fn([base_entropy, base_variance], base_ims)

    # TODO: generalize to any level energies, fusion strategy
    level_fusion_fn = pipeline_utils.get_fusion_fn(args.st_level_fusion, args)
    fused_layers = [{'laplacian': level_fusion_fn([pyramid.levels[level]['local_region'] for pyramid in stack.pyramids],
                                    [pyramid.levels[level]['laplacian'] for pyramid in stack.pyramids])
                    }
                    for level in range(stack.depth - 1)]
    fused_layers.append({'laplacian':fused_base})

    reconstruction_pyramid = LaplacianPyramid(levels=fused_layers, kernel=low_pass_kernel)
    fused_im = reconstruction_pyramid.reconstruct()
    return fused_im

def run_wavelet_pipeline(args, registered_images):
    def _fuse_single_color_ch(registered_images, ch_index):
        sub_band_names = ['LH', 'HL', 'HH']
        ims = [im[...,ch_index].astype(np.float32)/255. for im in registered_images]
        stack = StackPyramid(images=ims,
                depth=args.st_depth,
                kernel=None,
                pyramid_type='wavelet',
                wvt_name=args.st_wavelet_name
                )

        for energy_name in args.st_level_energies:
            fn = pipeline_utils.get_energy_fn(energy_name, args)
            for pyramid in stack.pyramids:
                for level in range(pyramid.depth):
                    for sub_band in sub_band_names:
                        pyramid.compute_energy_for_levels(level,
                                im_key=sub_band,
                                energy_fn=fn,
                                energy_key=f'{sub_band}_{energy_name}',
                                im_color=False,
                                )
        for energy_name in args.st_base_energies:
            fn = pipeline_utils.get_energy_fn(energy_name, args)
            for pyramid in stack.pyramids:
                pyramid.compute_energy_for_levels(-1,
                        im_key='image',
                        energy_fn=fn,
                        energy_key=energy_name,
                        im_color=False,
                        )

        fused_sub_bands = {sub_band: [] for sub_band in sub_band_names}
        for level_id in range(stack.depth):
            for sub_band in sub_band_names:
                energy_name = args.st_level_energies[0]
                input_energy = [pyramid.levels[level_id][f'{sub_band}_{energy_name}'] for pyramid in stack.pyramids]
                input_ims = [pyramid.levels[level_id][sub_band][...,np.newaxis] for pyramid in stack.pyramids]
                fused_sub_band = fusion.fuse_max(input_energy, input_ims)
                fused_sub_bands[sub_band].append(fused_sub_band)
        energy_name = args.st_base_energies[0]
        fused_base = fusion.fuse_max([pyramid.levels[-1][energy_name] for pyramid in stack.pyramids],
                                        [pyramid.levels[-1]['image'].astype(np.float32)[...,np.newaxis] for pyramid in stack.pyramids])

        recon_levels = [{sub_band: fused_sub_bands[sub_band][i][...,0] for sub_band in sub_band_names} for i in range(stack.depth)]
        recon_levels.append({'image': fused_base[...,0]})
        recon_pyramid = WaveletPyramid(levels=recon_levels, wvt_name=args.st_wavelet_name)
        fused_ch = recon_pyramid.reconstruct()
        return fused_ch

    fused_channels = []
    for ch_index in range(3):
        fused_channels.append(_fuse_single_color_ch(registered_images, ch_index))
    print(fused_channels)
    fused_im = np.stack(fused_channels,axis=-1)
    fused_im = np.clip(fused_im, 0, 1)
    fused_im = (fused_im * 255).astype(np.uint8)
    return fused_im


