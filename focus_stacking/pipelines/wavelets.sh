#!/bin/bash

python run_pipeline.py \
    --data.obj snail \
    --data.view 000 \
    --data.fnum f4.0 \
    --register.load \
    --pipeline scale_transform \
    --scale_transform.type wavelet \
    --scale_transform.wavelet_name db16 \
    --scale_transform.depth 5 \
    --scale_transform.base_energies local_entropy local_variance \
    --scale_transform.base_fusion multi_max \
    --scale_transform.level_energies squared \
    --scale_transform.level_fusion max \
    --energy.variance.neighborhood 9 \
    --energy.entropy.neighborhood 9 \
    --aux_out_dir /n/fs/3d-indoor/tmp_outdir/aux
