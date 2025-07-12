import numpy as np


def normalize_img(array, percentile=None, zero_centered=True, verbose=False):
    """
    Normalize an input array to [0, 1] or [-1, 1] using a specified percentile.

    Parameters
    ----------
    array : np.ndarray
        Input array to be normalized.
    percentile : float or None, optional
        Percentile value to use for the upper bound of normalization. If None,
        uses the maximum value of the array. Default is None.
    zero_centered : bool, optional
        If True, scales the normalized array to [-1, 1]. If False, keeps it in [0, 1].
        Default is True.
    verbose : bool, optional
        If True, prints the original and normalized value ranges. Default is False.

    Returns
    -------
    array : np.ndarray
        The normalized array.
    """
    # Compute minimum value of the array
    min_ = np.min(array)
    # Compute maximum value using percentile if provided, else use max
    if percentile is not None:
        max_ = np.percentile(array, percentile)
    else:
        max_ = np.max(array)
    # Optionally print the original range
    if verbose:
        print("original range: {},{}".format(min_, max_))

    # Normalize to [0, 1] if possible
    if max_ - min_ > 0:
        array = (array - min_) / (max_ - min_)
    # Optionally shift to [-1, 1]
    if zero_centered:
        array = array * 2 - 1
    # Optionally print the normalized range
    if verbose:
        print("normalized to range {}, {}".format(np.min(array), np.max(array)))
    return array


def renormalize_img(img, id_="", verbose=False):
    """
    Rescale an image array to the [0, 1] range.

    Parameters
    ----------
    img : np.ndarray
        Input image array to be rescaled.
    id_ : str, optional
        Identifier string for verbose output. Default is "".
    verbose : bool, optional
        If True, prints the rescaling information. Default is False.

    Returns
    -------
    np.ndarray
        The rescaled image array in [0, 1] range, or the original if degenerate.
    """
    # Optionally print the rescaling info
    if verbose:
        print(
            "{} | rescale from [{} . {}] to [0,1]".format(
                id_, img.min(), img.max()
            )
        )
    # Rescale if possible, else return original
    if img.max() - img.min() > 0:
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def random_crop(return_dict, img_keys, crop_size, dimension):
    """
    Apply random cropping to all images/segmentations in return_dict for the
    given img_keys.

    Parameters
    ----------
    return_dict : dict
        Dictionary containing image data, segmentations, and metadata.
    img_keys : list
        List of keys in return_dict to crop.
    crop_size : int
        Size for cropping.
    dimension : int
        Number of spatial dimensions (2 or 3).
    """
    
    if dimension == 3:
        # 3D cropping
        sx, sy, sz = return_dict["A"].shape[1:]
        crange = crop_size // 2

        # Random center coordinates for cropping
        cx = (
            np.random.randint(crange, sx - crange)
            if sx > 2 * crange
            else crange
        )
        cy = (
            np.random.randint(crange, sy - crange)
            if sy > 2 * crange
            else crange
        )
        cz = (
            np.random.randint(crange, sz - crange)
            if sz > 2 * crange
            else crange
        )

        # Crop all images/segmentations in img_keys
        for k in img_keys:
            tmp = return_dict[k]
            if len(tmp.shape) == 4:
                # Shape: (C, X, Y, Z)
                return_dict[k] = tmp[
                    :,
                    cx - crange : cx + crange,
                    cy - crange : cy + crange,
                    cz - crange : cz + crange,
                ]
            elif len(tmp.shape) == 3:
                # Shape: (X, Y, Z)
                return_dict[k] = tmp[
                    cx - crange : cx + crange,
                    cy - crange : cy + crange,
                    cz - crange : cz + crange,
                ]
            else:
                raise NotImplementedError("Unexpected data shape for cropping.")
        return return_dict
        del tmp
    elif dimension == 2:
        # 2D cropping
        sx, sy = return_dict["A"].shape[1:]
        crange = crop_size // 2

        cx = (
            np.random.randint(crange, sx - crange)
            if sx > 2 * crange
            else crange
        )
        cy = (
            np.random.randint(crange, sy - crange)
            if sy > 2 * crange
            else crange
        )
        for k in img_keys:
            tmp = return_dict[k]
            if len(tmp.shape) == 3:
                # Shape: (C, X, Y)
                return_dict[k] = tmp[
                    :,
                    cx - crange : cx + crange,
                    cy - crange : cy + crange,
                ]
            elif len(tmp.shape) == 2:
                # Shape: (X, Y)
                return_dict[k] = tmp[
                    cx - crange : cx + crange, cy - crange : cy + crange
                ]
            else:
                raise NotImplementedError("Unexpected data shape for cropping.")
        return return_dict
    else:
        raise NotImplementedError("Only 2D or 3D cropping is supported.")