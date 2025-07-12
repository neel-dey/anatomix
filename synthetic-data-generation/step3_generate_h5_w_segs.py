import nibabel as nib
import numpy as np
import h5py
import glob
import os


def main(args):
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Gather and sort view1 and view2 file paths
    view1s = sorted(glob.glob(os.path.join(args.view1_dir, "*.nii.gz")))
    view2s = sorted(glob.glob(os.path.join(args.view2_dir, "*.nii.gz")))

    # Sanity checks
    assert len(view1s) == len(view2s), "Number of view1 and view2 files must match"
    assert len(view1s) > 0, "No view1 files found"

    n_total = len(view1s)
    n_val = args.val_count
    n_train = n_total - n_val

    # Create HDF5 file for training data
    h5_train_path = os.path.join(args.out_dir, "train_data.hdf5")
    h5_train = h5py.File(h5_train_path, "w")

    # Process training samples
    for i in range(n_train):
        if i % args.print_every == 0:
            print(f"Processing training sample {i}/{n_train}")
        # Create a group for each sample
        grp = h5_train.create_group("{:06}".format(i))
        # Load view1 and view2 images
        view1 = nib.load(view1s[i]).get_fdata().astype(np.uint8)
        view2 = nib.load(view2s[i]).get_fdata().astype(np.uint8)
        # Stack views along a new axis
        combo = np.stack((view1, view2), axis=0)
        # Construct segmentation path
        seg_filename = os.path.basename(view1s[i]).split("view1_")[1]
        segpath = os.path.join(args.seg_dir, seg_filename)
        seg = nib.load(segpath).get_fdata().astype(np.uint8)
        # Store in HDF5
        grp["img"] = combo
        grp["seg"] = seg

    h5_train.close()

    # Create HDF5 file for validation data
    h5_val_path = os.path.join(args.out_dir, "val_data.hdf5")
    h5_val = h5py.File(h5_val_path, "w")

    # Process validation samples
    for idx, i in enumerate(range(n_train, n_total)):
        if i % args.print_every == 0:
            print(
                f"Processing validation sample {idx}/{n_val} (global idx {i})"
            )
        # Create a group for each sample
        grp = h5_val.create_group("{:06}".format(i))
        # Load view1 and view2 images
        view1 = nib.load(view1s[i]).get_fdata().astype(np.uint8)
        view2 = nib.load(view2s[i]).get_fdata().astype(np.uint8)
        # Stack views along a new axis
        combo = np.stack((view1, view2), axis=0)
        # Construct segmentation path
        seg_filename = os.path.basename(view1s[i]).split("view1_")[1]
        segpath = os.path.join(args.seg_dir, seg_filename)
        seg = nib.load(segpath).get_fdata().astype(np.uint8)
        # Store in HDF5
        grp["img"] = combo
        grp["seg"] = seg

    h5_val.close()


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate HDF5 files with segmentations from synthesized NIfTI views."
    )
    parser.add_argument(
        "--view1_dir",
        type=str,
        default="synthesized_views/view1",
        help="Path to directory containing view1 NIfTI files",
    )
    parser.add_argument(
        "--view2_dir",
        type=str,
        default="synthesized_views/view2",
        help="Path to directory containing view2 NIfTI files",
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        default="label_ensembles",
        help="Path to directory containing segmentation NIfTI files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./h5_w_segs/",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=100,
        help="Number of validation samples (from the end)",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="Print progress every print_every samples",
    )
    args = parser.parse_args()
    main(args)
