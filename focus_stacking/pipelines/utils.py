import focus_stacking.energy.energy as energies
import focus_stacking.fusion.fusion as fusion
import cv2

ENERGIES = ['local_variance', 'local_entropy', 'local_region', 'laplacian', 'self', 'squared']

def get_energy_fn(name, args):
    if name == 'local_variance':
        return lambda im, im_color : energies.local_variance(im,
                                     neighborhood=args.energy_variance_neighborhood,
                                     im_color=im_color)
    elif name == 'local_entropy':
        return lambda im, im_color : energies.local_entropy(im,
                                    neighborhood=args.energy_entropy_neighborhood,
                                    im_color=im_color)
    elif name == 'local_region':
        kernel = cv2.getGaussianKernel(args.energy_region_kwidth, sigma=-1)
        return lambda im, im_color : energies.local_region(im,
                                     kernel=kernel,
                                     neighborhood=args.energy_region_neighborhood,
                                     im_color=im_color)
    elif name == 'squared':
        return energies.squared

def get_fusion_fn(name, args):
    if name == 'max':
        return lambda energy, sources, return_indices : fusion.fuse_max(energy, sources, return_indices)
    elif name == 'multi_max_mean':
        return lambda energies, sources : fusion.fuse_multiple_max_mean(energies, sources)
    elif name == 'multi_max_vote':
        return lambda energies, sources : fusion.fuse_multiple_max_vote(energies, sources)
    elif name == 'multi_max':
        return lambda energies, sources : fusion.fuse_multiple_max(energies, sources)
