import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

from anatomix.registration.convex_adam_utils import (
    apply_avg_pool3d,
    diffusion_regularizer,
    correlate,
    coupled_convex,
    inverse_consistency,
    MINDSSC,
)


def merge_features(
    use_mask,
    pred_fixed,
    pred_moving,
    mask_fixed,
    mask_moving,
    fixed_img,
    moving_img,
):
    """
    Merge MIND-SSC descriptors with network features.
    Use masks if provided.
    
    Parameters
    ----------
    use_mask : bool
        Whether to use masking for feature merging.
    pred_fixed : torch.Tensor
        Predicted features for fixed image.
    pred_moving : torch.Tensor
        Predicted features for moving image.
    mask_fixed : torch.Tensor
        Binary mask for fixed image.
    mask_moving : torch.Tensor
        Binary mask for moving image.
    fixed_img : torch.Tensor
        Fixed image.
    moving_img : torch.Tensor
        Moving image.
        
    Returns
    -------
    tuple
        (mind_fixed, mind_moving, pred_fixed, pred_moving) 
        pred_fixed and pred_moving contain merged MINDSSC and network features.
    """
    if use_mask:
        H, W, D = pred_fixed.shape[-3:]

        # replicate masking
        avg3 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.AvgPool3d(3, stride=1)
        ).cuda()
        
        # Compute masked MIND-SSC for fixed and moving images

        # This is how the masks are used in the original repo for MINDSSC
        # Instead of zeroing out the masked regions, they fill them in with
        # distance transforms.
        mask = (avg3(mask_fixed.view(1, 1, H, W, D)) > 0.9).float()
        _, idx = edt(
            (mask[0, 0, ::2, ::2, ::2] == 0).squeeze().cpu().numpy(),
            return_indices=True,
        )
        fixed_r = F.interpolate(
            (fixed_img[..., ::2, ::2, ::2].reshape(-1)[
                idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]
            ]).unsqueeze(0).unsqueeze(0),
            scale_factor=2,
            mode='trilinear'
        )
        fixed_r.view(-1)[
            mask.view(-1)!=0
        ] = fixed_img.reshape(-1)[mask.view(-1)!=0]

        mask = (avg3(mask_moving.view(1, 1, H, W, D)) > 0.9).float()
        _, idx = edt(
            (mask[0, 0, ::2, ::2, ::2]==0).squeeze().cpu().numpy(),
            return_indices=True
        )
        moving_r = F.interpolate(
            (moving_img[..., ::2, ::2, ::2].reshape(-1)[
                idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]
            ]).unsqueeze(0).unsqueeze(0),
            scale_factor=2,
            mode='trilinear'
        )
        moving_r.view(-1)[
            mask.view(-1)!=0
        ] = moving_img.reshape(-1)[mask.view(-1)!=0]

        # Generate MIND-SSC descriptors
        mind_fixed = MINDSSC(fixed_r.cuda(), 1, 2)
        mind_moving = MINDSSC(moving_r.cuda(), 1, 2)

        # Apply masks to network features
        pred_fixed = pred_fixed * mask_fixed[None, None, ...]
        pred_moving = pred_moving * mask_moving[None, None, ...]        

        # Concatenate MINDSSC and network features
        pred_fixed = torch.concatenate([mind_fixed, pred_fixed], dim=1)
        pred_moving = torch.concatenate([mind_moving, pred_moving], dim=1)

    else:
        # Generate MIND-SSC descriptors without masking
        mind_fixed = MINDSSC(fixed_img, 1, 2)
        mind_moving = MINDSSC(moving_img, 1, 2)

        # Concatenate MINDSSC and network features
        pred_fixed = torch.concatenate([mind_fixed, pred_fixed], dim=1)
        pred_moving = torch.concatenate([mind_moving, pred_moving], dim=1)
    
    return mind_fixed, mind_moving, pred_fixed, pred_moving


def run_stage1_registration(
    features_fix_smooth,
    features_mov_smooth,
    disp_hw,
    grid_sp,
    sizes,
    n_ch,
    ic
):
    """
    Run first stage of registration using correlation volumes and 
    convex optimization.
    
    Parameters
    ----------
    features_fix_smooth : torch.Tensor
        Smoothed features from fixed image.
    features_mov_smooth : torch.Tensor
        Smoothed features from moving image.
    disp_hw : int
        Half-width of displacement search space.
    grid_sp : int
        Grid spacing for optimization.
    sizes : tuple
        Image dimensions (H, W, D).
    n_ch : int
        Number of feature channels.
    ic : bool
        Whether to enforce inverse consistency.
        
    Returns
    -------
    torch.Tensor
        High resolution displacement field.
    """
    H, W, D = sizes

    # Compute correlation volume with SSD
    ssd, ssd_argmin = correlate(
        features_fix_smooth,
        features_mov_smooth,
        disp_hw,
        grid_sp,
        (H, W, D),
        n_ch,
    )

    # provide auxiliary mesh grid
    disp_mesh_t = F.affine_grid(
        disp_hw*torch.eye(3, 4).cuda().half().unsqueeze(0),
        (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1),
        align_corners=True
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    
    # Perform coupled convex optimization
    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H,W,D))
    
    # Handle inverse consistency if requested
    if ic:
        scale = torch.tensor(
            [
                H // grid_sp - 1,
                W // grid_sp - 1,
                D // grid_sp - 1,
            ]
        ).view(1, 3, 1, 1, 1).cuda().half()/2

        # Compute reverse correlation
        ssd_, ssd_argmin_ = correlate(
            features_mov_smooth,
            features_fix_smooth,
            disp_hw,
            grid_sp,
            (H, W, D),
            n_ch,
        )

        disp_soft_ = coupled_convex(
            ssd_,
            ssd_argmin_,
            disp_mesh_t,
            grid_sp,
            (H, W, D),
        )
        disp_ice, _ = inverse_consistency(
            (disp_soft / scale).flip(1),
            (disp_soft_ / scale).flip(1),
            iterations=15,
        )

        disp_hr = F.interpolate(
            disp_ice.flip(1) * scale * grid_sp,
            size=(H, W, D),
            mode='trilinear',
            align_corners=False
        )
    
    else:
        disp_hr=disp_soft

    return disp_hr


def create_warp(
    disp_hr,
    sizes,
    grid_sp_adam,
):
    """
    
    Parameters
    ----------
    disp_hr : torch.Tensor
        High resolution displacement field.
    sizes : tuple
        Image dimensions (H, W, D).
    grid_sp_adam : int
        Grid spacing for Adam optimization.
        
    Returns
    -------
    torch.nn.Sequential
        A optimizable displacement grid.
    """
    H, W, D = sizes

    # create optimisable displacement grid
    disp_lr = F.interpolate(
        disp_hr,
        size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
        mode='trilinear',
        align_corners=False,
    )

    net = nn.Sequential(
        nn.Conv3d(
            3,
            1,
            (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
            bias=False,
        )
    )
    net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp_adam
    
    return net


def run_instance_opt(
    disp_hr,
    features_fix,
    features_mov,
    grid_sp_adam,
    lambda_weight,
    sizes,
    selected_niter,
    selected_smooth,
    lr=1,
):
    """
    Run instance-specific optimization to refine registration.
    
    Parameters
    ----------
    disp_hr : torch.Tensor
        Initial high resolution displacement field.
    features_fix : torch.Tensor
        Features from fixed image.
    features_mov : torch.Tensor
        Features from moving image.
    grid_sp_adam : int
        Grid spacing for Adam optimization.
    lambda_weight : float
        Weight for diffusion regularization.
    sizes : tuple
        Image dimensions (H, W, D).
    selected_niter : int
        Number of optimization iterations.
    selected_smooth : int
        Kernel size for final smoothing (3 or 5).
    lr : float, optional
        Learning rate for Adam optimizer. Default is 1.
        
    Returns
    -------
    torch.Tensor
        Optimized high resolution displacement field.
    """
    H, W, D = sizes
    
    with torch.no_grad():
        patch_features_fix = F.avg_pool3d(
            features_fix, grid_sp_adam, stride=grid_sp_adam,
        )
        patch_features_mov = F.avg_pool3d(
            features_mov, grid_sp_adam, stride=grid_sp_adam,
        )

    # Create warp tensor and optimizer
    net = create_warp(
        disp_hr, (H, W, D), grid_sp_adam,
    ).cuda()
    
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr,
    ) #TODO: make hparam

    # Run Adam optimization with diffusion regularization and B-spline smoothing
    for _ in range(selected_niter):
        optimizer.zero_grad()
        
        disp_sample = apply_avg_pool3d(
            net[0].weight, kernel_size=3, num_repeats=3,
        ).permute(0, 2, 3, 4, 1)
        
        # Calculate regularization loss
        reg_loss = diffusion_regularizer(disp_sample, lambda_weight)

        scale = torch.tensor(
            [
                (H // grid_sp_adam - 1) / 2,
                (W // grid_sp_adam - 1) / 2,
                (D // grid_sp_adam - 1) / 2,
            ]
        ).cuda().unsqueeze(0)
        
        # TODO: figure out why this needs to be here as opposed to above
        # and outside the loop. This was not the case in the original repo.
        grid0 = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0).cuda(),
            (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
            align_corners=False,
        )
        
        # Apply displacement to grid
        grid_disp = grid0.view(-1,3).cuda().float()
        grid_disp += ((disp_sample.view(-1, 3)) / scale).flip(1).float()

        # Sample moving features using displaced grid
        patch_mov_sampled = F.grid_sample(
            patch_features_mov.float(),
            grid_disp.view(
                1,
                H // grid_sp_adam,
                W // grid_sp_adam,
                D // grid_sp_adam,
                3,
            ).cuda(),
            align_corners=False,
            mode='bilinear',
        )

        # Calculate feature matching cost
        sampled_cost = (
            patch_mov_sampled - patch_features_fix
        ).pow(2).mean(1) * 12

        loss = sampled_cost.mean()

        # Combine losses and optimize
        total_loss = loss + reg_loss
        total_loss.backward()

        optimizer.step()

    # Generate final displacement field
    fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
    disp_hr = F.interpolate(
        fitted_grid * grid_sp_adam,
        size=(H, W, D),
        mode='trilinear',
        align_corners=False,
    )

    # Apply final smoothing if requested
    if selected_smooth in [3, 5]:
        disp_hr = apply_avg_pool3d(disp_hr, selected_smooth, num_repeats=3)
        
    return disp_hr
