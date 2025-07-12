import numpy as np
import torch

from scipy.ndimage import affine_transform

from monai.transforms import (
    ScaleIntensityd,
    Compose,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandSimulateLowResolutiond,
    ThresholdIntensityd
)

# -----------------------------------------------------------------------------
# Helpers for step1, label generation:
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Spatial transforms (affine, cropping, padding, etc) helpers:

# Rotations:
    
def get_rotation_matrix(degrees):
    """
    Calculate the 3D rotation matrix for given rotation angles in degrees.
    
    Parameters
    ----------
    degrees : array-like
        A list or array of three elements representing the rotation angles 
        in degrees.
    
    Returns
    -------
    matrix : numpy.ndarray
        A 3x3 numpy array representing the combined rotation matrix.
    """

    radians = np.radians(degrees)
    
    matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(radians[0]), -np.sin(radians[0])],
        [0, np.sin(radians[0]), np.cos(radians[0])]
    ])

    matrix_y = np.array([
        [np.cos(radians[1]), 0, np.sin(radians[1])],
        [0, 1, 0],
        [-np.sin(radians[1]), 0, np.cos(radians[1])]
    ])

    matrix_z = np.array([
        [np.cos(radians[2]), -np.sin(radians[2]), 0],
        [np.sin(radians[2]), np.cos(radians[2]), 0],
        [0, 0, 1]
    ])
    
    matrix = matrix_x @ matrix_y @ matrix_z
    
    return matrix


# Affine warping:

def apply_random_affine_transform(
    volume,
    rscale=0.5,
    rrotation=180,
    rtranslation=5,
    rshear=0.5,
    mode='nearest',
):
    """
    Apply a random affine transformation to a 3D volume.
    
    Parameters:
    -----------
    volume : ndarray
        The 3D volume to be transformed.
    rscale : float, optional
        The range for the random scaling factors. The scaling factors are 
        sampled uniformly from [1-rscale, 1+rscale]. Default is 0.5.
    rrotation : float, optional
        The range (in degrees) for the random rotation angles. The rotation 
        angles are sampled uniformly from [-rrotation, rrotation] for each 
        axis. Default is 180.
    rtranslation : float, optional
        The range for the random translation offsets. The translation offsets 
        are sampled uniformly from [-rtranslation, rtranslation] for each 
        axis. Default is 5.
    rshear : float, optional
        The range for the random shearing factors. The shearing factors are 
        sampled uniformly from [-rshear, rshear] for the shear matrix elements 
        above the diagonal. Default is 0.5.
    mode : str, optional
        The mode parameter determines how the input array is extended beyond 
        its boundaries. Default is 'nearest'.
    
    Returns:
    --------
    transformed_volume : ndarray
        The transformed 3D volume.
    """
    
    # Generate random affine transformation parameters
    scale = np.random.uniform(1. - rscale, 1. + rscale, 3)
    rotation = np.random.uniform(-rrotation, rrotation, 3)  # degrees
    translation = np.random.uniform(-rtranslation, rtranslation, 3)
    shear = np.random.uniform(-rshear, rshear, 3)
    reflection = np.random.choice([True, False], 3)
    
    # Build the affine matrices
    scale_matrix = np.diag(scale)
    rotation_matrix = get_rotation_matrix(rotation)
    shear_matrix = np.eye(3)
    shear_matrix[np.triu_indices(3, k=1)] = shear
    
    # Apply reflection
    for i in range(3):
        if reflection[i]:
            scale_matrix[i, i] *= -1
    
    # Combine the matrices to get the final affine matrix
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = scale_matrix @ rotation_matrix @ shear_matrix
    affine_matrix[:3, 3] = translation
    
    # Apply the affine transformation to the volume
    transformed_volume = affine_transform(
        volume, affine_matrix, mode=mode, cval=0.0, order=0,
    )
    
    return transformed_volume


# Cropping and padding:

def crop_and_pad_3d_volume(volume, target_size):
    """
    Crop a 3D volume to a non-zero bounding box and pad to the specified size.

    Parameters:
    -----------
    volume: ndarray
        A 3D numpy array representing the input volume.
    target_size: tuple 
        A 3-tuple of int (depth, height, width) specifying the desired size
        of the output volume after padding.

    Returns:
    --------
    padded volume: ndarray
        A 3D numpy array of the cropped and padded volume with the
        specified target size.

    """

    # Find non-zero indices along each axis
    non_zero_indices = np.where(volume != 0)

    # Determine the bounding box
    min_d, max_d = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    min_h, max_h = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_w, max_w = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])

    # Crop the volume to the bounding box
    cropped_volume = volume[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]

    # Pad the cropped volume to the target size
    pads = []
    pads.append(max(0, target_size[0] - cropped_volume.shape[0]))
    pads.append(max(0, target_size[1] - cropped_volume.shape[1]))
    pads.append(max(0, target_size[2] - cropped_volume.shape[2]))

    padding_tuples = []
    for j in range(len(pads)):
        if pads[j] % 2 == 0:
            padding_tuples.append((pads[j] // 2, pads[j] // 2))
        else:
            padding_tuples.append((pads[j] // 2 + 1, pads[j] // 2))

    padded_volume = np.pad(
        cropped_volume,
        (padding_tuples[0], padding_tuples[1], padding_tuples[2]),
        mode='constant',
    )

    return padded_volume


# -----------------------------------------------------------------------------
# Foreground sphere generator:


def generate_voxel_sphere(radius, array_shape, center_shift=None):
    """
    Generate a 3D numpy array representing a voxelized sphere.

    Parameters:
    -----------
    radius: float: 
        The radius of the sphere.
    array_shape: tuple of ints 
        The shape of the 3D array (depth, height, width).
    center_shift: tuple of ints, optional 
        A shift applied to the sphere's center within the array. 
        If not provided, the sphere is centered in the middle of the array.

    Returns:
    --------
    voxel_array: ndarray
        A 3D array with the voxelized sphere.

    """

    # Create an empty 3D array
    voxel_array = np.zeros(array_shape, dtype=int)
    
    # Calculate the center of the sphere
    center = np.array(array_shape) // 2
    if center_shift is not None:
        center = center + center_shift
    
    # Generate coordinates for all voxels in the array
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(array_shape[0]),
        np.arange(array_shape[1]),
        np.arange(array_shape[2]),
        indexing='ij'
    )
    
    # Calculate distances for all voxels to the center
    distances = np.sqrt(
        (x_coords - center[0]) ** 2 +
        (y_coords - center[1]) ** 2 +
        (z_coords - center[2]) ** 2
    )
    
    # Set values within the sphere's radius to 1
    voxel_array[distances <= radius] = 1
    
    return voxel_array


def draw_perlin_deformation(
    out_shape,
    scales,
    min_std=0,
    max_std=1,
    dtype=torch.float32,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    #TODO: merge draw_perlin_volume and draw_perlin_deformation
    
    Generates a multi-scale Perlin noise tensor with specified characteristics,
    as defined in https://arxiv.org/abs/2004.10282
    
    Used for generating a random warp to apply to a 3D volume.
    
    Parameters
    ----------
    out_shape : tuple or list of int
        The shape of the output tensor.
    scales : int or list of int
        Scale factors for generating Perlin noise. If an integer is provided, 
        a single scale is used. If a list, multiple scales are combined.
    min_std : float, optional
        The minimum standard deviation for the Gaussian noise. Default 0.
    max_std : float, optional
        The maximum standard deviation for the Gaussian noise. Default 1.
    dtype : torch.dtype, optional
        The desired data type of the output tensor. Default torch.float32.
    device : torch.device, optional
        The device on which to create the tensor. Default CUDA if available, 
        otherwise CPU.
    
    Returns
    -------
    torch.Tensor
        A tensor of shape `out_shape` containing the generated Perlin noise.
    
    """
    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    out = torch.zeros(tuple(out_shape), dtype=dtype, device=device)

    for scale in scales:
        sample_shape = np.ceil(out_shape[1:] / scale)
        sample_shape = np.int32((out_shape[0], *sample_shape))

        std = (max_std - min_std) * torch.rand(
            (1,), dtype=torch.float32, device=device
        )
        std = std + min_std
        gauss = std * torch.randn(
            tuple(sample_shape), dtype=torch.float32, device=device
        )

        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        if scale == 1:
            out += gauss
        else:
            out += torch.nn.functional.interpolate(
                gauss[None, ...],
                scale_factor=zoom[1:],
                mode='trilinear'
            )[0]

    return out


def rescale_coords(arrsize):
    """
    Rescale coordinates to be in the range [-1, 1].
    
    Parameters
    ----------
    arrsize : int
        The size of the array for which to generate rescaled coordinates.

    Returns
    -------
    np.ndarray
        A numpy array of rescaled coordinates in the range [-1, 1].

    """
    
    return 2 * (np.arange(arrsize) - ((arrsize - 1) / 2)) / (arrsize - 1)


def generate_grid_unit(imgshape):
    """
    Taken from 
    https://github.com/cwmok/LapIRN/blob/master/Code/Functions.py#L18
    
    Generate a grid of unit coordinates for a given image shape.
    
    Parameters
    ----------
    imgshape : tuple of int
        The shape of the image for which to generate the grid. 
        Should be a 3-tuple (x, y, z).

    Returns
    -------
    np.ndarray
        A numpy array representing the grid of unit coordinates with shape 
        (imgshape[0], imgshape[1], imgshape[2], 3).
    
    """
    x = rescale_coords(imgshape[0])
    y = rescale_coords(imgshape[1])
    z = rescale_coords(imgshape[2])
    
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)

    return grid


def sample_corruption(
    grid,
    arrsize=(128, 128, 128),
    min_std=1.,
    max_std=5.,
    scales=(8, 16, 32),
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Generates a warped binary sphere by applying Perlin noise-like deformation.

    Parameters
    ----------
    grid : torch.Tensor
        The grid tensor to which the deformation will be applied.
    arrsize : tuple of int, optional
        The size of the 3D array representing the volume. 
        Default is (128, 128, 128).
    min_std : float, optional
        The minimum standard deviation for the Perlin noise. Default is 1.0.
    max_std : float, optional
        The maximum standard deviation for the Perlin noise. Default is 5.0.
    scales : tuple of int, optional
        The scales at which to generate Perlin noise. Default is (8, 16, 32).
    device : torch.device, optional
        The device on which to perform the computation (CPU or CUDA). 
        Default is CUDA if available, otherwise CPU.

    Returns
    -------
    defsphere : torch.Tensor
        The deformed sphere as a tensor.
    """
    # Sample a random radius and center for the foreground sphere
    # This range of radii and centers was chosen arbitrarily
    radius = np.random.randint(48, 72)
    new_center = tuple(np.random.randint(-32, 32, size=3))
    
    # Generate a binary sphere with the sampled radius and center:
    initial_arr = np.abs(
        1 - generate_voxel_sphere(radius, arrsize, new_center)
    )
    initial_arr = torch.from_numpy(initial_arr.astype(np.float32)).to(device)
    
    # Given a binary sphere, we now use it to get a foreground mask with an
    # arbitrary shape. We will use SynthMorph-style deformations to do so.

    # Let's sample noise to deform the binary sphere:
    randdef = draw_perlin_deformation(
        out_shape=(3,) + arrsize,
        scales=scales,
        min_std=min_std,
        max_std=max_std,
    )
    
    randdef[0] = 2 * randdef[0] / (float(arrsize[0]) - 1)
    randdef[1] = 2 * randdef[1] / (float(arrsize[1]) - 1)
    randdef[2] = 2 * randdef[2] / (float(arrsize[2]) - 1)
    
    # Deform the sphere:
    defsphere = torch.nn.functional.grid_sample(
        initial_arr[None, None, ...],
        grid[None, ...] + randdef.permute(1, 2, 3, 0)[None, ...],
        mode='nearest',
        padding_mode='reflection',
    )

    return defsphere


# -----------------------------------------------------------------------------
# Helpers for step2, paired volume generation:
# -----------------------------------------------------------------------------


def get_transforms():
    """
    Generates a MONAI composed transformation set for augmenting the GMM
    sampled intensities. As we are pretraining contrastively, we sample two
    contrastive views/volumes per each 3D labelmap ("view1" and "view2" below).
    These are then augmented as below.

    See the comments below to walk through the transforms.

    Returns
    -------
    train_transforms : monai.transforms.Compose
        A MONAI Compose object containing the specified sequence of transforms.

    Notes
    -----
    The specific probabilities and parameter ranges for each transformation are
    based on the empirical settings used in the paper. Play around with it!
    """
    
    train_transforms = Compose(
        [
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["view1", "view2"]),
            # Apply bias fields:
            RandBiasFieldd(
                keys=["view1"], prob=0.98, coeff_range=(0.0, 0.075),
            ),
            RandBiasFieldd(
                keys=["view2"], prob=0.98, coeff_range=(0.0, 0.075),
            ),
            # Apply K-spikes:
            RandKSpaceSpikeNoised(keys=["view1"], prob=0.2),
            RandKSpaceSpikeNoised(keys=["view2"], prob=0.2),
            # Apply gamma transforms:
            RandAdjustContrastd(keys=["view1"], prob=0.5, gamma=(0.5, 2.)),
            RandAdjustContrastd(keys=["view2"], prob=0.5, gamma=(0.5, 2.)),
            # Apply smoothing:
            RandGaussianSmoothd(
                keys=["view1"],
                prob=0.5,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            RandGaussianSmoothd(
                keys=["view2"],
                prob=0.5,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            # Apply gibbs ringing (applies a box mask to kspace. alpha=0, box
            # width=1, i.e. no masking. alpha=1, boxwidth=0, i.e. all masked):
            RandGibbsNoised(keys=["view1"], prob=0.5, alpha=(0.0, 0.333)),
            RandGibbsNoised(keys=["view2"], prob=0.5, alpha=(0.0, 0.333)),
            # Apply sharpening:
            RandGaussianSharpend(keys=["view1"], prob=0.25),
            RandGaussianSharpend(keys=["view2"], prob=0.25),
            # Simulate much bigger voxels. MONAI does it nnUNet style, as
            # opposed to TorchIO's (IMO better) per-axis anisotropic style:
            RandSimulateLowResolutiond(keys=["view1"], prob=0.333),
            RandSimulateLowResolutiond(keys=["view2"], prob=0.333),
            # Clip out negative values:
            ThresholdIntensityd(
                keys=["view1", "view2"], above=True, threshold=0.,
            ),
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["view1", "view2"]),
        ]
    )
    return train_transforms


def draw_perlin_volume(
    out_shape,
    scales,
    min_std=0,
    max_std=1,
    dtype=torch.float32,
    device="cpu",
):
    """
    #TODO: merge draw_perlin_volume and draw_perlin_deformation
    
    Generates a 3D tensor with Perlin noise as defined in
    https://arxiv.org/abs/2004.10282

    Parameters
    ----------
    out_shape : tuple of int
        Shape of the output tensor (e.g., (D, H, W)).
    scales : float or list of float
        List of scales at which to generate the noise. 
        A single float can also be provided.
    min_std : float, optional
        Minimum standard deviation of the Gaussian noise. Default is 0.
    max_std : float, optional
        Maximum standard deviation of the Gaussian noise. Default is 1.
    dtype : torch.dtype, optional
        Data type of the output tensor. Default is torch.float32.
    device : str or torch.device, optional
        Device on which to create the tensor. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape `out_shape` with generated Perlin noise.
    """
    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    out = torch.zeros(tuple(out_shape), dtype=dtype, device=device)

    for scale in scales:
        sample_shape = np.ceil(out_shape / scale).astype(np.uint8)
    
        std = (max_std - min_std) * torch.rand(
            (1,), dtype=torch.float32, device=device
        )
        std = std + min_std
        gauss = std * torch.randn(
            tuple(sample_shape), dtype=torch.float32, device=device
        )
    
        zoom = [o // s for o, s in zip(out_shape, sample_shape)]
        if scale == 1:
            out += gauss
        else:
            out += torch.nn.functional.interpolate(
                gauss[None, None, ...],
                scale_factor=scale,
                mode='trilinear'
            )[0, 0, ...]

    return out


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def sample_gmm(means, stds, label_map, zero_bckgnd=0.25):
    """
    Generate a synthetic image using a Gaussian Mixture Model (GMM).
    
    This function creates a synthetic image where each region corresponding 
    to a unique label in the 3D synthetic `label_map` is filled with values
    from a Gaussian distribution characterized by the specified means and 
    standard deviations (`stds`). 
    
    100*zero_bckgnd % of the time, fill background label with zeros.

    Parameters
    ----------
    means : list or np.ndarray
        A list or array of means for the Gaussians, one for each label.
    stds : list or np.ndarray
        A list or array of std devs for the Gaussians, one for each label.
    label_map : np.ndarray
        A 3D array where each element corresponds to a label indicating the
        region in the synthetic image.
    zero_bckgnd : float
        Probability of filling background with zeros instead of intensities.

    Returns
    -------
    torch.Tensor
        A synthetic image/torch Tensor with values generated from the Gaussian 
        distributions, with values clipped to a minimum of 0 and scaled using 
        min-max normalization.
        
    """
    labels = np.unique(label_map)
    synthimage = torch.zeros(label_map.shape, requires_grad=False)

    for i, label in enumerate(labels):
        if (i == 0) and (torch.rand(1) < zero_bckgnd):
            continue
        indices = label_map==label
        synthimage[indices] = stds[i] * torch.randn(indices.sum()) + means[i]

    synthimage = torch.clip(synthimage, min=0)
    synthimage = minmax(synthimage)

    return synthimage


def transform_uniform(arr, minval, maxval):
    """
    Transform arr from a uniform distribution in [0, 1] to [minval, maxval].
    """
    assert arr.min() >= 0
    assert arr.max() <= 1
    return (maxval - minval) * arr + minval