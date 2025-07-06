import os, subprocess, sys
import argparse, glob
import focus_stacking.focus_stack.__main__ as pyramid_stack

# data dir format is /n/fs/3d-indoor/macro_data/focus_stack/{obj}/{view}/{fnum}/images/*.JPG 

METHODS_ROOT='/n/fs/3d-indoor/focus_stacking/focus_stacking/methods'
DATA_ROOT='/n/fs/3d-indoor/macro_data/focus_stacks'

# Read current existing methods
with open(f'{METHODS_ROOT}/methods.txt') as f:
    methods = f.read().splitlines()
methods = [f'{i}: {methods[i]}' for i in range(len(methods))]
objs = [os.path.basename(d) for d in glob.glob(f'{DATA_ROOT}/*')]
help_msg = 'Current supported methods:\n' + '\n'.join(methods)

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--obj', choices=objs, type=str, help='name of object')
parser.add_argument('--view', type=str, help='view dir, formatted as 03:d')
parser.add_argument('--fnum', choices=['f4.0', 'f5.6', 'f6.3', 'f13', 'f22'], type=str, help='aperture size')
parser.add_argument('--imdir', type=str)

#m1 = parser.add_argument_group('m1 args')
subparsers = parser.add_subparsers(dest='method')

m1 = subparsers.add_parser('m1', help='1: focus-stacking (pyramids)')
m1.add_argument('--choice', choices=['pyramid', 'average', 'max'], type=str)
m1.add_argument('--energy', choices=['laplacian', 'sobel'], type=str)
m1.add_argument('--min_size', type=int)
m1.add_argument('--kernel_size', type=int)
m1.add_argument('--blur_size', type=int)
m1.add_argument('--smooth_size', type=int)
m1.add_argument('--align_iters', type=int)
m1.add_argument('--align_eps', type=float)


m2 = subparsers.add_parser('m2', help='2: focus-stack (wavelets)')
m2.add_argument('--no_crop', action='store_true')
m2.add_argument('--reference', type=int)
m2.add_argument('--global_align', action='store_true')
m2.add_argument('--full_resolution_align', action='store_true')
m2.add_argument('--no_whitebalance', action='store_true')
m2.add_argument('--no_contrast', action='store_true')
m2.add_argument('--align_keep_size', action='store_true')
m2.add_argument('--consistency', type=int)
m2.add_argument('--denoise', type=float)




args = parser.parse_args()

if args.imdir is None:
    WORKING_PATH=f'{DATA_ROOT}/{args.obj}/{args.view}/{args.fnum}'
    IMGS_PATH=f'{WORKING_PATH}/images'
    RESULTS_PATH=f'{WORKING_PATH}/results'
    os.makedirs(RESULTS_PATH, exist_ok=True)
else:
    WORKING_PATH='/n/fs/3d-indoor/focus_stacking'
    IMGS_PATH=args.imdir
    RESULTS_PATH=f'{WORKING_PATH}/results'
    os.makedirs(RESULTS_PATH, exist_ok=True)

if args.method == 'm1':
    # TODO: compute ECC ONLY on resized images
    # currently resizes entire input->output
    pyramid_stack.main(src_dir=IMGS_PATH,
                       dest_dir=RESULTS_PATH,
                       choice=args.choice,
                       energy=args.energy,
                       pyramid_min_size=args.min_size,
                       kernel_size=args.kernel_size,
                       blur_size=args.blur_size,
                       smooth_size=args.smooth_size,
                       align_iters=args.align_iters,
                       align_eps=args.align_eps,)

if args.method == 'm2':
    output_fname = 'm2'
    run_cmd = [f'focus-stack',
              ]
    if args.no_crop:
        run_cmd.append('--nocrop')
        output_fname += '_no-crop'
    if args.reference is not None:
        run_cmd.append(f'--reference={args.reference}')
        output_fname += f'_ref-{args.reference}'
    if args.global_align:
        run_cmd.append('--global-align')
        output_fname += '_ga'
    if args.full_resolution_align:
        run_cmd.append('--full-resolution-align')
        output_fname += '_fa'
    if args.no_whitebalance:
        run_cmd.append('--no-whitebalance')
        output_fname += '_nw'
    if args.no_contrast:
        run_cmd.append('--no-contrast')
        output_fname += '_nw'
    if args.align_keep_size:
        run_cmd.append('--align-keep-size')
        output_fname += '_align-keep-size'
    if args.consistency:
        run_cmd.append(f'--consistency={args.consistency}')
        output_fname += f'_c-{args.consistency}'
    if args.denoise:
        run_cmd.append(f'--denoise={args.denoise}')
        output_fname += f'_d-{args.denoise}'
    output_fname += '.jpg'
    run_cmd.append(f'--output={RESULTS_PATH}/{output_fname}')
    run_cmd.append(f'{IMGS_PATH}/*')
    run_cmd = ' '.join(run_cmd)
    print(run_cmd)
    subprocess.run(run_cmd, shell=True)

