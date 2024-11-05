import torch
import numpy as np
import os
import random
import argparse
import nibabel as nib

from glob import glob
from concurrent.futures import ProcessPoolExecutor

from datagen_utils import (
    get_transforms,
    sample_gmm,
    draw_perlin_volume,
    transform_uniform,
)


# -----------------------------------------------------------------------------
# generate images:

def process_volume(
    lab,
    volshape,
    means_range,
    stds_range,
    perl_scales,
    perl_max_std,
    perl_mult_factor,
    transforms,
    savedir,
    seed,
):
    """
    Process a single volume: load the label ensemble, generate synthetic views,
    apply transformations, and save the outputs.
    
    Parameters
    ----------
    lab : str
        Path to the label ensemble nifti file.
    volshape : tuple of int
        Shape of the volume.
    means_range : tuple of int
        Range of means for the Gaussian distributions.
    stds_range : tuple of int
        Range of standard deviations for the Gaussians.
    perl_scales : tuple of int
        Scales for generating Perlin-like noise.
    perl_max_std : float
        Maximum standard deviation for Perlin-like noise.
    perl_mult_factor : float
        Multiplicative constant applied to sampled Perlin-like noise.
    train_transforms : monai.transforms.Compose
        A MONAI Compose object containing the specified sequence of transforms.
    savedir : str
        Directory where the output synthetic volumes/views will be saved.
    seed : int
        Random seed for the process.
    
    Returns
    -------
    None
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(
        'Synthesizing ensemble {} with seed {}'.format(
            os.path.basename(lab), seed,
        )
    )
    
    # Initialize MONAI augmentation pipeline:
    transforms = get_transforms()

    current_label = nib.load(lab).get_fdata()
    labels = np.unique(current_label)

    # Sample random means and std devs for both paired volumes:
    means1 = transform_uniform(
        torch.rand(len(labels)), means_range[0], means_range[1],
    )
    means2 = transform_uniform(
        torch.rand(len(labels)), means_range[0], means_range[1],
    )
    
    stds1 = transform_uniform(
        torch.rand(len(labels)), stds_range[0], stds_range[1],
    )
    stds2 = transform_uniform(
        torch.rand(len(labels)), stds_range[0], stds_range[1],
    )

    # Sample two volumes from the specified GMMs:
    synthview1 = sample_gmm(means1, stds1, current_label)
    synthview2 = sample_gmm(means2, stds2, current_label)

    # Sample Perlin-like noise to simulate spatial structure in texture:
    randperl_view1 = draw_perlin_volume(
        out_shape=volshape,
        scales=perl_scales,
        max_std=perl_max_std,
    )
    
    randperl_view2 = draw_perlin_volume(
        out_shape=volshape,
        scales=perl_scales,
        max_std=perl_max_std,
    )

    # Pointwise multiply with Perlin-like noise and downscale intensities 
    # by `perl_mult_factor`:
    synthperl1 = synthview1 * (1 + perl_mult_factor * randperl_view1)
    synthperl2 = synthview2 * (1 + perl_mult_factor * randperl_view2)
    
    # Create data dict and send to MONAI augmentation pipeline:
    inputimgs = {
        "view1": synthperl1, "view2": synthperl2, "label": current_label,
    }
    outputs = transforms(inputimgs)

    # Save synthetic volumes as nifti files:
    # Saving as uint8 volumes to not blow up disk usage.
    nib.save(
        nib.Nifti1Image(
            (outputs['view1'] * 255.).astype(np.uint8),
            affine=np.eye(4)
        ),
        '{}/view1/view1_{}'.format(savedir, os.path.basename(lab)),
    )
    
    nib.save(
        nib.Nifti1Image(
            (outputs['view2'] * 255.).astype(np.uint8),
            affine=np.eye(4),
        ),
        '{}/view2/view2_{}'.format(savedir, os.path.basename(lab)),
    )


def run(
    idx_start,
    idx_end,
    savedir,
    label_fpaths='./label_ensembles/',
    volshape=(128, 128, 128),
    means_range=(25, 255),
    stds_range=(5, 20),
    perl_scales=(4, 8, 16, 32),
    perl_max_std=5.,
    perl_mult_factor=0.02,
    max_workers=None,
):
    """
    Generate paired synthetic volumes from pre-generated label ensembles using
    the appearance model from the paper.
    
    Given a 3D label map, we sample means and standard deviations at random
    from which to sample initial intensities. For paired volume generation, we
    sample two sets of moments. These initial volumes are then extensively
    augmented to create highly variable training volumes.
    
    Takes a path to pre-generated label files, creates a list of label files,
    and synthesizes nifti files for indices between `idx_start` and `idx_end`.

    Parameters
    ----------
    idx_start : int
        The starting index of the label files to process.
    idx_end : int
        The ending index of the label files to process.
    savedir : str
        Directory where the output synthetic views will be saved.
    label_fpaths : str, optional
        Path to the directory containing label files. 
        Default ./label_ensembles/'.
    volshape : tuple of int, optional
        Shape of the volume. Default (128, 128, 128).
    means_range : tuple of int, optional
        Range of means for the Gaussian distributions. Default [25, 255].
    stds_range : tuple of int, optional
        Range of standard deviations for the Gaussians. Default [5, 20].
    perl_scales : tuple of int, optional
        Scales for generating Perlin noise. Default (4, 8, 16, 32).
    perl_max_std : float, optional
        Maximum standard deviation for Perlin noise. Default 5.0.
    perl_mult_factor : float, optional
        Multiplication factor for Perlin noise. Default 0.02.
    max_workers : int, optional
        Maximum number of worker processes to use. 
        Default is None, which uses all available resources.

    Returns
    -------
    None
    """
    # Load list of precomputed label ensembles:
    labs = sorted(glob(label_fpaths + '/*.nii.gz'))
    assert len(labs) > 0

    # Initialize MONAI augmentation pipeline:
    transforms = get_transforms()

    # Generate random seeds for each process
    random_seeds = np.random.randint(
        1234, high=100000000, size=len(labs[idx_start:idx_end])
    ).tolist()

    # Process volumes in parallel:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_volume,
                lab,
                volshape,
                means_range,
                stds_range,
                perl_scales,
                perl_max_std,
                perl_mult_factor,
                transforms,
                savedir,
                seed
            )
            for lab, seed in zip(labs[idx_start:idx_end], random_seeds)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--start_idx', type=int, default=0,
        help='Starting index of list of label ensemble files to process',
    )
    parser.add_argument(
        '--end_idx', type=int, default=120000,
        help='Ending index of list of label ensemble files to process',
    )
    parser.add_argument(
        '--ensembledir', type=str, default='./label_ensembles/',
        help='Path to where the synthetic label ensembles are saved'
    )
    parser.add_argument(
        '--savedir', type=str, default='./synthesized_views/',
        help='Path to save synthesized volumes to'
    )
    parser.add_argument(
        '--max_workers', type=int, default=3,
        help='Maximum number of worker processes to use',
    )

    args = parser.parse_args()

    os.makedirs('{}/view1/'.format(args.savedir), exist_ok=True)
    os.makedirs('{}/view2/'.format(args.savedir), exist_ok=True)

    run(
        args.start_idx,
        args.end_idx,
        savedir=args.savedir,
        label_fpaths=args.ensembledir,
        max_workers=args.max_workers
    )
