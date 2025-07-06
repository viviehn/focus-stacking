import os, sys
import argparse, cv2, logging, pdb
import focus_stacking.registration.ecc as ecc
import focus_stacking.utils.images as im_utils
import focus_stacking.utils.data_loader as data_loader
from focus_stacking.utils.pyramids import StackPyramid, Pyramid
import focus_stacking.utils.pyramids
import focus_stacking.energy.energy as energies
import focus_stacking.fusion.fusion as fusion
import focus_stacking.pipelines.scale_transform_pipeline as scale_transform
import focus_stacking.pipelines.simple_pipeline as simple_pipeline

align_max_iters = 50
align_eps = 1e-3
DATA_ROOT = '/n/fs/3d-indoor/macro_data/focus_stacks'

ENERGIES = ['local_variance', 'local_entropy', 'local_region', 'laplacian', 'self', 'squared']
FUSION_STRATEGIES = ['max', 'multi_max']
SCALE_TRANSFORMS = ['laplacian_pyramid', 'wavelet']
PIPELINES = ['simple', 'scale_transform']

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data.obj', type=str, help='top level object directory', dest='obj')
    parser.add_argument('--data.view', type=str, help='view dir, formatted as 03:d', dest='view')
    parser.add_argument('--data.fnum', choices=['f4.0', 'f5.6', 'f6.3', 'f13', 'f22'], type=str, help='aperture size', dest='fnum')
    parser.add_argument('--data.src_dir', type=str, help='manually specify full path to image source directory', dest='src_dir')

    parser.add_argument('--register.load', action='store_true', help='if true, load registered images from aux dir and skip alignment', dest='load_reg_images')
    parser.add_argument('--register.save_images', action='store_true', help='write registered images to aux output directory', dest='reg_save_im')
    parser.add_argument('--register.ref_im_idx', type=int, help='reference image to align image to', dest='ref_im_idx', default=0)
    parser.add_argument('--register.res', type=int, help='downscale image by [res] for warping only', dest='warp_res', default=1)
    parser.add_argument('--register.max_iters', type=int, help='max iterations for ECC', dest='align_max_iters', default=50)
    parser.add_argument('--register.eps', type=float, help='termination eps for ECC', dest='align_eps', default=1e-3)

    parser.add_argument('--pipeline', type=str, choices=PIPELINES)

    parser.add_argument('--simple.energies', nargs='+', choices=ENERGIES, dest='simple_energies')
    parser.add_argument('--simple.fusion', choices=FUSION_STRATEGIES, dest='simple_fusion')

    parser.add_argument('--scale_transform.type', type=str, choices=SCALE_TRANSFORMS, help='energies used for pyramid base fusion', dest='st_type')
    parser.add_argument('--scale_transform.wavelet_name', type=str, help='energies used for pyramid base fusion', dest='st_wavelet_name')
    parser.add_argument('--scale_transform.kwidth', type=int, help='kernel width for pyramid construction', default=63, dest='st_kwidth')
    parser.add_argument('--scale_transform.depth', type=int, help='pyramid depth/number of levels in pyramid', default=5, dest='st_depth')
    parser.add_argument('--scale_transform.base_energies', nargs='+', choices=ENERGIES, help='energies used for pyramid base fusion', dest='st_base_energies')
    parser.add_argument('--scale_transform.base_fusion', choices=FUSION_STRATEGIES, help='energies used for pyramid base fusion', dest='st_base_fusion')
    parser.add_argument('--scale_transform.level_energies', nargs='+', choices=ENERGIES, help='energies used for pyramid base fusion', dest='st_level_energies')
    parser.add_argument('--scale_transform.level_fusion',  choices=FUSION_STRATEGIES, help='energies used for pyramid base fusion', dest='st_level_fusion')

    parser.add_argument('--energy.variance.neighborhood', type=int, dest='energy_variance_neighborhood')
    parser.add_argument('--energy.entropy.neighborhood', type=int, dest='energy_entropy_neighborhood')
    parser.add_argument('--energy.region.neighborhood', type=int, dest='energy_region_neighborhood')
    parser.add_argument('--energy.region.kwidth', type=int, dest='energy_region_kwidth')
    parser.add_argument('--energy.laplacian.neighborhood', type=int)
    parser.add_argument('--energy.laplacian.kwidth', type=int)
    
    parser.add_argument('--out_dir', type=str, help='manually specify output directory')
    parser.add_argument('--out_fname', type=str, default='fused_im.jpg')
    parser.add_argument('--aux_out_dir', type=str, help='manually specify auxiliary output directory')
    parser.add_argument('--results_out_dir', type=str, help='manually specify results output directory')

    #subparsers = parser.add_subparsers(dest='method')

    args = parser.parse_args()

    

    src_dir = data_loader.get_src_dir(args, DATA_ROOT)
    out_dir = os.path.dirname(src_dir)
    
    if src_dir is None:
        print('No valid source dir was found.')
        sys.exit()
    aux_out_dir = f'{out_dir}/aux' if args.aux_out_dir is None else args.aux_out_dir
    results_out_dir = f'{out_dir}/results' if args.results_out_dir is None else args.results_out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(aux_out_dir, exist_ok=True)
    os.makedirs(results_out_dir, exist_ok=True)
    logging.basicConfig(filename=f'{out_dir}/run_pipeline.log', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.info('Started')

    images = data_loader.load_images_from_dir(src_dir)
    gray_images = [im_utils.bgr_to_grayscale(im) for im in images]
    if args.load_reg_images:
        registered_images = data_loader.load_images_from_dir(aux_out_dir, match='reg_*.jpg')
    else:
        gray_registered_images, registered_images = ecc.align(gray_images, images, args.ref_im_idx, args.warp_res, args.align_max_iters, args.align_eps,
                                                              save_img_dir=aux_out_dir if args.reg_save_im else None)

    
    if args.pipeline == 'simple':
        fused_im = simple_pipeline.run_pipeline(args, registered_images)
        
    elif args.pipeline == 'scale_transform':
        if args.st_type == 'laplacian_pyramid':
            fused_im = scale_transform.run_laplacian_pyramid_pipeline(args, registered_images)
        if args.st_type == 'wavelet':
            fused_im = scale_transform.run_wavelet_pipeline(args, registered_images)

    cv2.imwrite(f'{results_out_dir}/{args.out_fname}', fused_im)

    # fuse base



    '''
    for idx, pyramid in enumerate(down_pyramid.pyramids):
        pyramid.display_pyramid(f'{aux_out_dir}/pyr_gauss_{idx}.jpg', f'{aux_out_dir}/pyr_lap_{idx}.jpg')

    energy_list = []
    for image in registered_images:
        #energy = energies.local_variance(image,ksize=9)
        energy = energies.local_entropy(image,ksize=9)
        energy_list.append(energy)

    # fusion

    

    fused_im = fusion.max(energy_list, registered_images)


    # Pyramid based methods
    '''

    '''

    # fusion
    '''

