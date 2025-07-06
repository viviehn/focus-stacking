import numpy as np

# Fuse image by taking max value per slice of energy stack
def fuse_max(energy, images):
    energy_stack = np.stack(energy)
    image_stack = np.stack(images)
    max_idx = np.argmax(energy_stack, axis=0)
    fused_im = np.choose(max_idx[:,:,np.newaxis],image_stack)
    return fused_im

def fuse_min(energy, images):
    energy_stack = np.stack(energy)
    image_stack = np.stack(images)
    max_idx = np.argmin(energy_stack, axis=0)
    fused_im = np.choose(max_idx[:,:,np.newaxis],image_stack)
    return fused_im

# energies should be a list of lists
def fuse_multiple_max(energies, images):
    num_energies = len(energies)
    energy_fused = [fuse_max(e, images) for e in energies]
    energy_fused = np.stack(energy_fused)
    fused_im = np.mean(energy_fused,axis=0)
    return fused_im
