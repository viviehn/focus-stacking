import numpy as np

def _fuse_by_index(indices, source_stack):
    print(indices.shape, source_stack.shape)
    fused_im = np.choose(indices[:,:,np.newaxis],source_stack)
    return fused_im

def _get_indices(energy_stack, get='max'):
    if get == 'max':
        return np.argmax(energy_stack, axis=0)
    elif get == 'min':
        return np.argmin(energy_stack, axis=0)
    
# Fuse image by taking max value per slice of energy stack
def fuse_max(energy, source, return_indices=False):
    energy_stack = np.stack(energy)
    source_stack = np.stack(source)
    max_idx = _get_indices(energy_stack)
    if return_indices:
        return _fuse_by_index(max_idx, source_stack), max_idx
    else:
        return _fuse_by_index(max_idx, source_stack)

def fuse_min(energy, source):
    energy_stack = np.stack(energy)
    source_stack = np.stack(source)
    min_idx = _get_indices(energy_stack, get='min')
    return _fuse_by_index(min_idx, source_stack)

# energies should be a list of lists
def fuse_multiple_max_mean(energies, sources):
    fused_per_energy = []
    for e, src in zip(energies, sources):
        fused = fuse_max(e, src)
        fused_per_energy.append(fused)
    fused_im = np.mean(np.stack(fused_per_energy),axis=0)
    return fused_im

# 2 out of 3 voting
def fuse_multiple_max_vote(energies, sources, return_individual=True):
    fused_per_energy = []
    indices_per_energy = []
    stacked_sources = []
    for e, src in zip(energies, sources):
        energy_stack = np.stack(e)
        max_indices = _get_indices(energy_stack)
        indices_per_energy.append(max_indices)
        stacked_sources.append(np.stack(src))
    
    # perform voting
    # if at least num_vote energies agree on an index
    # set the third energy to that index as well
    
    new_indices_per_energy = []
    for idx in range(3):
        new_indices_per_energy.append(indices_per_energy[idx].copy())
        vote_idx = np.where(indices_per_energy[(idx+1)%3] == indices_per_energy[(idx+2)%3])
        new_indices_per_energy[idx][vote_idx] = indices_per_energy[(idx+1)%3][vote_idx]
        
    for indices, src in zip(new_indices_per_energy, stacked_sources):
        fused_per_energy.append(_fuse_by_index(indices, src))
        
    if return_individual:
        return fused_per_energy
    else:   
        fused_im = np.mean(np.stack(fused_per_energy),axis=0)
        return fused_im
    
# 2 out of 3 voting
def fuse_multiple_max(energies, sources, return_individual=True):
    fused_per_energy = []
    indices_per_energy = []
    stacked_sources = []
    for e, src in zip(energies, sources):
        energy_stack = np.stack(e)
        max_indices = _get_indices(energy_stack)
        indices_per_energy.append(max_indices)
        stacked_sources.append(np.stack(src))
        
    for indices, src in zip(indices_per_energy, stacked_sources):
        fused_per_energy.append(_fuse_by_index(indices, src))
        
    if return_individual:
        return fused_per_energy
    else:   
        fused_im = np.mean(np.stack(fused_per_energy),axis=0)
        return fused_im