import numpy as np
import nibabel as nib
import os
import argparse
import concurrent.futures

from glob import glob


# -----------------------------------------------------------------------------
# Preprocessing label helpers:

def delete_ct_images(basedir):
    """
    Delete CT intensity volume files from the given base directory. We never
    use the actual volumes, only the segmentations.

    Parameters
    ----------
    basedir : str
        The base directory containing the CT intensity volume files to delete.
    
    Returns
    -------
    None
    """
    cts = sorted(glob(basedir + '/**/ct.nii.gz'))
    
    # In case you've run this script before:
    if len(cts) == 0:
        return
    
    # Delete CT volumes:
    print("Deleting CT intensity volumes...")
    for ct in cts:
        os.remove(ct)
    print("Done deleting.")
    

def merge_vertebrae_and_ribs_worker(segdir):
    """
    Merge individual rib and vertebra label files in a segmentation directory.

    Parameters
    ----------
    segdir : str
        The folder containing individual rib and vertebra label nifti files.
    
    Returns
    -------
    None
    """

    print(f"Merging labels in {segdir}")
    fpaths = sorted(glob(segdir + '/*.nii.gz'))
    
    dummy_for_metadata = nib.load(fpaths[0])
    
    # Empty arrays for aggregating labels:
    rib_labels = np.zeros(dummy_for_metadata.shape)
    vert_labels = np.zeros(dummy_for_metadata.shape)

    for fpath in fpaths:
        segdata = nib.load(fpath).get_fdata()
        
        if segdata.sum() == 0:
            # Totalsegmentator includes blank label files for structures that
            # aren't present. Let's remove them.
            os.remove(fpath)
            continue
        else:
            # Merge labels. They dont overlap and can be added directly.
            if 'rib_' in os.path.basename(fpath):
                rib_labels += segdata
            if 'vertebrae_' in os.path.basename(fpath):
                vert_labels += segdata

    # Save new merged label files if they aren't blank:
    if rib_labels.sum() > 0:
        nib.save(
            nib.Nifti1Image(
                rib_labels.astype(np.uint8),
                dummy_for_metadata.affine,
            ),
            segdir + '/all_ribs.nii.gz',
        )

    if vert_labels.sum() > 0:
        nib.save(
            nib.Nifti1Image(
                vert_labels.astype(np.uint8),
                dummy_for_metadata.affine,
            ),
            segdir + '/all_vertebrae.nii.gz',
        )
        

def merge_vertebrae_and_ribs(basedir, max_workers=None):
    """
    Merge rib and vertebra labels in all segmentation directories in the base
    directory.

    Parameters
    ----------
    basedir : str
        The base directory containing the segmentation directories.
    max_workers : int, optional
        The maximum number of worker processes to use.
        Default is None, i.e., use all available CPU cores.
    
    Returns
    -------
    None
    """
    segdirs = sorted(glob(basedir + '/**/segmentations/'))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        executor.map(merge_vertebrae_and_ribs_worker, segdirs)


def delete_individual_vertebrae_and_ribs(basedir):
    """
    Delete individual rib and vertebra label files from all segmentation
    directories in the given base directory.

    Parameters
    ----------
    basedir : str
        The base directory containing the segmentation directories.
    
    Returns
    -------
    None
    """
    ribs = sorted(glob(basedir + '/**/segmentations/rib_*.nii.gz'))
    verts = sorted(glob(basedir + '/**/segmentations/vertebrae_*.nii.gz'))

    rmfiles = ribs + verts
    
    print("Deleting individualized rib and vertebral label files")
    for rmfile in rmfiles:
        os.remove(rmfile)
        

# -----------------------------------------------------------------------------
# Main script:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--totalsegmentator_path',
        type=str,
        default='./Totalsegmentator_dataset/',
        help='Path to unzipped TotalSegmentator v1 dataset',
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum number of worker processes to use',
    )

    args = parser.parse_args()
    
    # Delete CT volumes:
    delete_ct_images(args.totalsegmentator_path)
    
    # Merge ribs and vertebrae labels:
    merge_vertebrae_and_ribs(
        args.totalsegmentator_path,
        args.max_workers,
    )
    
    # Clean up individual label files:
    delete_individual_vertebrae_and_ribs(args.totalsegmentator_path)
    
    print("All ready to synthesize label ensembles.")
