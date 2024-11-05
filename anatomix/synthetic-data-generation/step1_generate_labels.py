import numpy as np
import nibabel as nib
import os
import torch
import argparse
import random
import string

import torch.multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor
from glob import glob
from skimage.filters import median
from skimage import morphology

from datagen_utils import (
    crop_and_pad_3d_volume,
    apply_random_affine_transform,
    sample_corruption,
    generate_grid_unit,
)

# -----------------------------------------------------------------------------
# generate 3D label volumes:

def generate_label_ensemble(
    segs, grid, idx, min_shapes, max_shapes, savedir, sidelen=128, seed=None,
):
    """
    Generate a 3D label ensemble volume with random shapes and transformations.
    This function is called repeatedly in parallel to speed things up.
    
    Parameters
    ----------
    segs : list of str
        List of file paths to the segmentation templates.
    grid : torch.Tensor
        Grid unit used for sampling deformation applied to foreground mask.
    idx : int
        Index of the current label ensemble being generated.
    min_shapes : int
        Minimum number of shapes to include in the ensemble.
    max_shapes : int
        Maximum number of shapes to include in the ensemble.
    savedir : str
        Directory where the generated label ensemble will be saved.
    sidelen : int, optional
        Side length of the generated volumes. Default 128.
    seed : int, optional
        Random seed for reproducibility. Default None.
        
    Returns
    -------
    None
    """

    # Set a unique seed for each process
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print('Synthesizing ensemble {:05d} with seed {}'.format(idx + 1, seed))
    identifier = 'unconstrained'
    n_templates = np.random.randint(min_shapes, max_shapes)
    label_ensemble = np.zeros((sidelen, sidelen, sidelen), dtype=np.uint8)
    
    # Sample and transform templates
    for k in range(n_templates):
        # TotalSegmentator has a bunch of empty label files.
        # To handle segmentation files that are empty, we just check to
        # make sure that their sum is nonzero.
        
        # Load templates:
        template = np.zeros(1)
        while template.sum() == 0:
            template = nib.load(
                np.random.choice(segs)
            ).get_fdata().astype(np.uint8)

        # Foreground crop and pad to `sidelen` along each axis:
        croptemplate = crop_and_pad_3d_volume(
            template, (sidelen, sidelen, sidelen),
        )
                    
        # Apply random affine transfrormation to the cropped template
        croptemplate = apply_random_affine_transform(
            croptemplate, mode='grid-wrap',
        )
        
        # Sample (sidelen, sidelen, sidelen) ROI:
        roitemplate = croptemplate[:sidelen, :sidelen, :sidelen]
        
        # Assign template a label corresponding to the sampling iteration:
        label_ensemble[roitemplate > 0] = k * 1

    # Median smooth:
    label_ensemble = median(label_ensemble)

    # 2/3rds of the time, apply a foreground mask:
    if np.random.rand() > 0.33333:
        identifier = 'foreground_masked'
        
        # Generate a foreground mask. This mask starts off as a sphere with a
        # random center and radius and is then randomly deformed.
        # The 5 here could have been randomized too. You should try it :)
        deformedsphere = ~sample_corruption(grid, max_std=5.).type(torch.bool) 
        deformedspherenp = deformedsphere.cpu().numpy().squeeze()
        deformedspherenp = median(deformedspherenp)
        
        # Apply mask and increment all labels as we've now created a new
        # background:
        label_ensemble = deformedspherenp * label_ensemble
        label_ensemble[deformedspherenp > 0] += 1
        
        # 1/3rd of the time, create an envelope around the foreground mask:
        if np.random.rand() > 0.5:
            identifier = 'foreground_masked_enveloped'
            
            # Sample variable widths for the envelope:
            envelopekernel = np.random.choice([2, 3, 4])
            
            # Morphological ops to create the envelope/rim/etc.:
            dilsphere = morphology.dilation(
                deformedspherenp, footprint=morphology.ball(envelopekernel),
            )
            erosphere = morphology.erosion(
                deformedspherenp, footprint=morphology.ball(envelopekernel),
            )
            
            envelopesphere = np.logical_and(
                dilsphere, ~erosphere,
            ).astype(np.uint8)
        
            # Assign new label to newly synthesized envelope region:
            label_ensemble[envelopesphere > 0] = 1 + label_ensemble.max()
    
    print('chose {}'.format(identifier))
    
    # Sample a filename for the synthesized volume. We use a random ASCII
    # string as a unique suffix to just keep the files separate as the 
    # actual name is <type_of_label>_<numberoftemplates>, which is
    # non-identifiable.
    randstr = ''.join(
        random.choices(string.ascii_uppercase + string.digits, k=7)
    )
    
    fpath = os.path.join(savedir, '{}_shapes{}_{}.nii.gz'.format(
        identifier, n_templates, randstr,
    ))

    # The edge case where the <type_of_label>_<numberoftemplates>_<randomascii>
    # name convention has a collision is annoying so just redraw the ASCII:
    while os.path.isfile(fpath):
        randstr = ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=7)
        )
        fpath = os.path.join(savedir, '{}_shapes{}_{}.nii.gz'.format(
            identifier, n_templates, randstr,
        ))
    
    nib.save(
        nib.Nifti1Image(label_ensemble, np.eye(4)),
        fpath,
    )


def main(
    segs,
    grid,
    n_vols,
    min_shapes,
    max_shapes,
    savedir,
    sidelen=128,
    max_workers=None,
):
    """
    Synthesize random 3D label ensembles in parallel.
    
    Parameters
    ----------
    segs : list of str
        List of file paths to the segmentation templates.
    grid : torch.Tensor
        Grid unit used for sampling corruption.
    n_vols : int
        Number of volumes to generate.
    min_shapes : int
        Minimum number of shapes to include in each ensemble.
    max_shapes : int
        Maximum number of shapes to include in each ensemble.
    savedir : str
        Directory where the generated label ensembles will be saved.
    sidelen : int, optional
        Side length of the generated volumes. Default 128.
    max_workers : int, optional
        Maximum number of workers for parallel processing.
        Default None, corresponding to using all workers.
        
    Returns
    -------
    None
    """
    mp.set_start_method("spawn")

    # Generate random seeds for each process
    seeds = np.random.choice(
        range(1, n_vols*10), size=n_vols, replace=False,
    )
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                generate_label_ensemble, 
                segs, 
                grid, 
                idx, 
                min_shapes, 
                max_shapes, 
                savedir, 
                sidelen, 
                seeds[idx]
            )
            for idx in range(n_vols)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 3D label ensembles')
    parser.add_argument(
        '--n_ensembles', type=int, default=120000,
        help='Number of 3D label ensemble volumes to generate',
    )
    parser.add_argument(
        '--min_templates', type=int, default=20,
        help='Minimum number of shapes to include in each ensemble',
    )
    parser.add_argument(
        '--max_templates', type=int, default=40,
        help='Maximum number of shapes to include in each ensemble',
    )
    parser.add_argument(
        '--side_length', type=int, default=128,
        help='Side length of the generated volumes',
    )
    parser.add_argument(
        '--templatedir', type=str, default='./Totalsegmentator_dataset/',
        help='Path to unzipped and preprocessed TotalSegmentator data',
    )
    parser.add_argument(
        '--savedir', type=str, default='./label_ensembles/',
        help='Directory to save the generated label ensembles',
    )
    parser.add_argument(
        '--max_workers', type=int, default=None,
        help='Maximum number of workers for parallel processing',
    )

    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)

    # TODO: could make this less hardcoded for template sets other than
    # TotalSegmentator
    fpaths = glob(
        args.templatedir + '/**/segmentations/*.nii.gz',
        recursive=True,
    )
    
    # Create a Torch grid that'll be used for deformations later:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = torch.from_numpy(
        generate_grid_unit(
            (args.side_length, args.side_length, args.side_length),
        ).astype(np.float32)
    ).to(device)
    grid.requires_grad = False
    
    # Run
    main(
        fpaths,
        grid,
        args.n_ensembles,
        args.min_templates,
        args.max_templates,
        args.savedir,
        max_workers=args.max_workers
    )
