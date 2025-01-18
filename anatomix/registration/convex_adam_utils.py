import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import time

from monai.inferers import sliding_window_inference

from anatomix.model.network import Unet


def load_model(ckpt_path):
    """
    Load a pretrained U-Net model from a checkpoint file.
    
    Parameters
    ----------
    ckpt_path : str
        Path to the model checkpoint file.
        
    Returns
    -------
    model : torch.nn.Module
        Loaded U-Net model in frozen evaluation mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(3, 1, 16, 4, ngf=16).to(device)

    model.to(device)
    model.load_state_dict(torch.load(ckpt_path), strict=True)
        
    model.eval()
    return model


def diffusion_regularizer(disp_sample, lambda_weight):
    """
    Calculate diffusion regularization loss for a displacement field.
    
    Parameters
    ----------
    disp_sample : torch.Tensor
        Displacement field tensor of shape [1, C, H, W, D].
    lambda_weight : float
        Weight factor for the regularization loss.
        
    Returns
    -------
    float
        Weighted regularization loss value.
    """
    loss = (
        ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() +
        ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() +
        ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
    )
    return lambda_weight * loss


def apply_avg_pool3d(disp_hr, kernel_size, num_repeats):
    """
    Apply 3D average pooling multiple times to a displacement field.
    
    Parameters
    ----------
    disp_hr : torch.Tensor
        High resolution displacement field.
    kernel_size : int
        Size of the pooling kernel.
    num_repeats : int
        Number of times to apply the pooling operation.
        
    Returns
    -------
    torch.Tensor
        Pooled displacement field.
    """
    padding = kernel_size // 2
    for _ in range(num_repeats):
        disp_hr = F.avg_pool3d(
            disp_hr, 
            kernel_size, 
            padding=padding, 
            stride=1
        )
    return disp_hr


def minmax(arr, minclip=None, maxclip=None):
    """
    Normalize array to [0,1] range with optional clipping.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Input array to normalize.
    minclip : float, optional
        Minimum value for clipping.
    maxclip : float, optional
        Maximum value for clipping.
        
    Returns
    -------
    numpy.ndarray
        Normalized array in [0,1] range.
    """
    if not (minclip is None) & (maxclip is None):
        arr = np.clip(arr, minclip, maxclip)
        
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def extract_features(
    img_fixed,
    img_moving,
    model,
    fixminclip=None,
    fixmaxclip=None,
    movminclip=None,
    movmaxclip=None,
):
    """
    Extract features from fixed and moving images using pretrained network.
    
    Parameters
    ----------
    img_fixed : numpy.ndarray
        Fixed image.
    img_moving : numpy.ndarray
        Moving image.
    model : torch.nn.Module
        Neural network model for feature extraction.
    fixminclip : float, optional
        Minimum clip value for fixed image.
    fixmaxclip : float, optional
        Maximum clip value for fixed image.
    movminclip : float, optional
        Minimum clip value for moving image.
    movmaxclip : float, optional
        Maximum clip value for moving image.
        
    Returns
    -------
    tuple
        (fixed_features, moving_features) extracted by the model.
    """
    imfixed = minmax(img_fixed, fixminclip, fixmaxclip)
    imfixed = torch.from_numpy(imfixed)[None, None, ...].float().cuda()
    imfixed.requires_grad = False

    immoving = minmax(img_moving, movminclip, movmaxclip)
    immoving = torch.from_numpy(immoving)[None, None, ...].float().cuda()
    immoving.requires_grad = False
    
    with torch.no_grad():
        opfixed = sliding_window_inference(
            imfixed,
            (128, 128, 128),
            2,
            model,
            overlap=0.8,
            mode="gaussian",
            sigma_scale=0.25,
        )
        opmoving = sliding_window_inference(
            immoving,
            (128, 128, 128),
            2,
            model,
            overlap=0.8,
            mode="gaussian",
            sigma_scale=0.25,
        )
    
    return opfixed, opmoving

# -----------------------------------------------------------------------------
# JacDet utils

def generate_grid(imgshape):
    """
    Generate a coordinate grid for an image volume.
    
    Parameters
    ----------
    imgshape : tuple
        Shape of the image volume (H, W, D).
        
    Returns
    -------
    numpy.ndarray
        Coordinate grid of shape (H, W, D, 3).
    """
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def JacobianDet(y_pred, sample_grid):
    """
    Calculate the Jacobian determinant of a deformation field.
    
    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted deformation field.
    sample_grid : torch.Tensor
        Reference coordinate grid.
        
    Returns
    -------
    torch.Tensor
        Jacobian determinant of the deformation.
    """
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (
        dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1]
    )
    Jdet1 = dx[:,:,:,:,1] * (
        dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0]
    )
    Jdet2 = dx[:,:,:,:,2] * (
        dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0]
    )

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def pdist_squared(x):
    """
    Compute pairwise squared Euclidean distances between points.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of points.
        
    Returns
    -------
    torch.Tensor
        Matrix of pairwise squared distances.
    """
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


# -----------------------------------------------------------------------------
# Original Convex Adam utils


def MINDSSC(img, radius=2, dilation=2):
    """
    See http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details 
    on the MIND-SSC descriptor
    
    Parameters
    ----------
    img : torch.Tensor
        Input image tensor.
    radius : int, optional
        Radius for neighborhood sampling.
    dilation : int, optional
        Dilation factor for sampling pattern.
        
    Returns
    -------
    torch.Tensor
        MIND-SSC descriptor tensor.
    """
    # Kernel size
    kernel_size = radius * 2 + 1
    
    # Define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 2],
        [2, 1, 1],
        [1, 2, 1]
    ]).long()
    
    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(
        1, 6, 1,
    ).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(
        6, 1, 1,
    ).view(-1, 3)[mask, :]
    
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[
        torch.arange(12) * 27 +
        idx_shift1[:, 0] * 9 +
        idx_shift1[:, 1] * 3 +
        idx_shift1[:, 2]
    ] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift2.view(-1)[
        torch.arange(12) * 27 +
        idx_shift2[:, 0] * 9 +
        idx_shift2[:, 1] * 3 +
        idx_shift2[:, 2]
    ] = 1

    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    
    # compute patch-ssd
    img_padded = rpad1(img)
    conv1 = F.conv3d(img_padded, mshift1, dilation=dilation)
    conv2 = F.conv3d(img_padded, mshift2, dilation=dilation)
    ssd = F.avg_pool3d(
        rpad2((conv1 - conv2) ** 2),
        kernel_size,
        stride=1,
    )
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(
        mind_var,
        mind_var.mean().item() * 0.001,
        mind_var.mean().item() * 1000,
    )
    mind /= mind_var
    mind = torch.exp(-mind)
    
    #permute to have same ordering as C++ code
    mind = mind[
        :,
        torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(),
        :,
        :,
        :,
    ]
    
    return mind


def correlate(mind_fix, mind_mov, disp_hw, grid_sp, shape, ch=12):
    """
    Compute correlation between fixed and moving MIND descriptors.
    
    Parameters
    ----------
    mind_fix : torch.Tensor
        MIND descriptor of fixed image.
    mind_mov : torch.Tensor
        MIND descriptor of moving image.
    disp_hw : int
        Half-width of displacement search range.
    grid_sp : int
        Grid spacing for displacement field.
    shape : tuple
        Shape of the input volume.
    ch : int, optional
        Number of MIND descriptor channels.
        
    Returns
    -------
    tuple
        (ssd, ssd_argmin) Sum of squared differences and its argmin.
    """
    H = int(shape[0])
    W = int(shape[1])
    D = int(shape[2])

    with torch.no_grad():
        mind_mov_padded = F.pad(
            mind_mov,
            (disp_hw, disp_hw, disp_hw, disp_hw, disp_hw, disp_hw)
        ).squeeze(0)
        mind_unfold = F.unfold(mind_mov_padded, disp_hw * 2 + 1)
        mind_unfold = mind_unfold.view(
            ch,
            -1,
            (disp_hw * 2 + 1) ** 2,
            W // grid_sp,
            D // grid_sp,
        )
 
    # Initialize SSD (Sum of Squared Differences) tensor and the argmin tensor
    ssd = torch.zeros(
        (disp_hw * 2 + 1) ** 3,
        H // grid_sp,
        W // grid_sp,
        D // grid_sp,
        dtype=mind_fix.dtype,
        device=mind_fix.device
    )
    ssd_argmin = torch.zeros(H // grid_sp, W // grid_sp, D // grid_sp).long()

    with torch.no_grad():
        for i in range(disp_hw * 2 + 1):
            mind_diff = (
                mind_fix.permute(1, 2, 0, 3, 4) -
                mind_unfold[:, i: i + H // grid_sp]
            )
            mind_sum = mind_diff.pow(2).sum(0, keepdim=True)
            ssd[i::(disp_hw * 2 + 1)] = apply_avg_pool3d(
                mind_sum.transpose(2,1), kernel_size=3, num_repeats=2,
            )
            
        # Reshape and permute the SSD tensor to the required format
        ssd = ssd.view(
            disp_hw * 2 + 1,
            disp_hw * 2 + 1,
            disp_hw * 2 + 1,
            H // grid_sp,
            W // grid_sp,
            D // grid_sp
        )
        ssd = ssd.transpose(1,0)
        ssd = ssd.reshape(
            (disp_hw * 2 + 1) ** 3,
            H // grid_sp,
            W // grid_sp,
            D // grid_sp
        )
        ssd_argmin = torch.argmin(ssd, 0)

    return ssd, ssd_argmin


def coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, shape):
    """
    Solve coupled convex optimization problems for global regularization.
    
    Parameters
    ----------
    ssd : torch.Tensor
        Sum of squared differences tensor.
    ssd_argmin : torch.Tensor
        Argmin of the SSD tensor.
    disp_mesh_t : torch.Tensor
        Displacement mesh template.
    grid_sp : int
        Grid spacing.
    shape : tuple
        Shape of the input volume.
        
    Returns
    -------
    torch.Tensor
        Optimized displacement field.
    """
    H = int(shape[0])
    W = int(shape[1])
    D = int(shape[2])

    disp_soft = F.avg_pool3d(
        disp_mesh_t.view(3, -1)[:, ssd_argmin.view(-1)].reshape(
            1, 3, H // grid_sp, W // grid_sp, D // grid_sp,
        ),
        kernel_size=3,
        padding=1,
        stride=1,
    )

    # Coefficients for the convex optimization
    coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)

        with torch.no_grad():
            for i in range(H//grid_sp):
                coupled = ssd[:, i, :, :]
                coupled += coeffs[j] * (
                    disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)
                ).pow(2).sum(0).view(-1, W // grid_sp, D // grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled,0)


        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3, -1)[:, ssd_coupled_argmin.view(-1)].reshape(
                1, 3, H // grid_sp, W // grid_sp, D // grid_sp,
            ),
            kernel_size=3,
            padding=1,
            stride=1,
        )
    return disp_soft


def inverse_consistency(disp_field1s,disp_field2s,iterations=20):
    """
    Enforce inverse consistency between forward and backward transforms.
    
    Parameters
    ----------
    disp_field1s : torch.Tensor
        Forward displacement field.
    disp_field2s : torch.Tensor
        Backward displacement field.
    iterations : int, optional
        Number of consistency iterations.
        
    Returns
    -------
    tuple
        (disp_field1i, disp_field2i) Inverse consistent displacement fields.
    """
    B, C, H, W, D = disp_field1s.size()
    
    # Make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        # Create an identity grid:
        identity = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0),
            (1, 1, H, W, D),
        ).permute(0, 4, 1, 2, 3)
    
        identity = identity.to(disp_field1s.device).to(disp_field1s.dtype)

        for _ in range(iterations):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5 * (disp_field1s - F.grid_sample(
                    disp_field2s,
                    (identity + disp_field1s).permute(0, 2, 3, 4, 1)
                )
            )
            disp_field2i = 0.5 * (disp_field2s - F.grid_sample(
                    disp_field1s,
                    (identity + disp_field2s).permute(0, 2, 3, 4, 1)
                )
            )

    return disp_field1i, disp_field2i
