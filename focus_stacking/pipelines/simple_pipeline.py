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
import focus_stacking.pipelines.utils as pipeline_utils


def run_pipeline(args, registered_images):
    transform = lambda x : x
    transformed_images = [transform(im) for im in registered_images]
    energy_fns = {}
    for energy_name in args.simple_energies:
        fn = pipeline_utils.get_energy_fn(energy_name, args)
        energy_fns[energy_name] = {'fn': fn}
        energy_fns[energy_name]['energies'] = []
    for image in transformed_images:
        for energy_name in energy_fns.keys():
            energy_fn = energy_fns[energy_name]['fn']
            e = energy_fn(image, True) # todo: image is color or not?
            energy_fns[energy_name]['energies'].append(e)
    
    fusion_fn = pipeline_utils.get_fusion_fn(args.simple_fusion, args)
    
    energies = [energy_fns[energy_name]['energies'] for energy_name in energy_fns.keys()]
    if len(energies) == 1:
        energies = energies[0]
        
    print(energies)
    fused_im, indices = fusion_fn(energies, registered_images, return_indices=True)
    return fused_im, indices