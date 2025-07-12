import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_tensor(format, save_path, save_name, data_dict, override=True):
    if format == "npz":
        save_dict = dict()
        import numpy as np
    os.makedirs(save_path, exist_ok=True)
    for k in data_dict:
        f = data_dict[k].cpu().detach().numpy()
        if format == "npz":
            save_dict[k] = f
            print(f.shape)
        if format == "nii":
            save_suffix = "%s_%s.nii.gz" % (save_name, k)
            if not os.path.exists(save_path + save_suffix) or override:
                import SimpleITK as sitk

                if len(f.shape) == 5:
                    img = sitk.GetImageFromArray(f[0].transpose(1, 2, 3, 0))
                elif len(f.shape) == 4:
                    img = sitk.GetImageFromArray(f.transpose(1, 2, 3, 0))
                else:
                    raise NotImplementedError
                print(
                    "{} Saving nifti {} - Image shape: {}, data type: {}".format(
                        k, save_suffix, f.shape, f.dtype
                    )
                )
                print(save_path)
                sitk.WriteImage(img, save_path + save_suffix)
            else:
                print("{} exists.".format(save_suffix))

    if format == "npz":
        print("Saving npz...")
        if not os.path.exists(save_path + "%s.npz" % save_name) or override:
            np.savez(save_path + "%s.npz" % save_name, **save_dict)
        else:
            print("File exists, skip.")
