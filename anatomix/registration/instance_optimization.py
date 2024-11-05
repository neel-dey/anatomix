import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

from convex_adam_utils import (
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
    fixed_ch0,
    moving_ch0,
):
    if use_mask:
        H, W, D = pred_fixed.shape[-3:]

        # replicate masking
        avg3 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.AvgPool3d(3, stride=1)
        ).cuda()
        
        mask = (avg3(mask_fixed.view(1, 1, H, W, D)) > 0.9).float()
        _, idx = edt(
            (mask[0, 0, ::2, ::2, ::2] == 0).squeeze().cpu().numpy(),
            return_indices=True,
        )
        fixed_r = F.interpolate(
            (fixed_ch0[..., ::2, ::2, ::2].reshape(-1)[
                idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]
            ]).unsqueeze(0).unsqueeze(0),
            scale_factor=2,
            mode='trilinear'
        )
        fixed_r.view(-1)[
            mask.view(-1)!=0
        ] = fixed_ch0.reshape(-1)[mask.view(-1)!=0]

        mask = (avg3(mask_moving.view(1, 1, H, W, D)) > 0.9).float()
        _, idx = edt(
            (mask[0, 0, ::2, ::2, ::2]==0).squeeze().cpu().numpy(),
            return_indices=True
        )
        moving_r = F.interpolate(
            (moving_ch0[..., ::2, ::2, ::2].reshape(-1)[
                idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]
            ]).unsqueeze(0).unsqueeze(0),
            scale_factor=2,
            mode='trilinear'
        )
        moving_r.view(-1)[
            mask.view(-1)!=0
        ] = moving_ch0.reshape(-1)[mask.view(-1)!=0]

        mind_fixed = MINDSSC(fixed_r.cuda(), 1, 2)
        mind_moving = MINDSSC(moving_r.cuda(), 1, 2)

        pred_fixed = pred_fixed * mask_fixed[None, None, ...]
        pred_moving = pred_moving * mask_moving[None, None, ...]        

        pred_fixed = torch.concatenate([mind_fixed, pred_fixed], dim=1)
        pred_moving = torch.concatenate([mind_moving, pred_moving], dim=1)

    else:
        mind_fixed = MINDSSC(fixed_ch0, 1, 2)
        mind_moving = MINDSSC(moving_ch0, 1, 2)
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
    H, W, D = sizes

    # compute correlation volume with SSD
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
    
    # perform coupled convex optimisation
    disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H,W,D))
    
    # if "ic" flag is set: make inverse consistent
    if ic:
        scale = torch.tensor(
            [
                H // grid_sp - 1,
                W // grid_sp - 1,
                D // grid_sp - 1,
            ]
        ).view(1, 3, 1, 1, 1).cuda().half()/2

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
    sizes,  # 3-tuple or list
    selected_niter,
    selected_smooth,
    lr=1,
):
    H, W, D = sizes
    
    with torch.no_grad():
        patch_features_fix = F.avg_pool3d(
            features_fix, grid_sp_adam, stride=grid_sp_adam,
        )
        patch_features_mov = F.avg_pool3d(
            features_mov, grid_sp_adam, stride=grid_sp_adam,
        )

    net = create_warp(
        disp_hr, (H, W, D), grid_sp_adam,
    ).cuda()
    
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr,
    ) #TODO: make hparam

    # run Adam optimisation with diffusion reg. and B-spline smoothing
    for _ in range(selected_niter):
        optimizer.zero_grad()
        
        disp_sample = apply_avg_pool3d(
            net[0].weight, kernel_size=3, num_repeats=3,
        ).permute(0, 2, 3, 4, 1)
        
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
        
        grid_disp = grid0.view(-1,3).cuda().float()
        grid_disp += ((disp_sample.view(-1, 3)) / scale).flip(1).float()

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

        sampled_cost = (
            patch_mov_sampled - patch_features_fix
        ).pow(2).mean(1) * 12

        loss = sampled_cost.mean()

        total_loss = loss + reg_loss
        total_loss.backward()

        optimizer.step()


    fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
    disp_hr = F.interpolate(
        fitted_grid * grid_sp_adam,
        size=(H, W, D),
        mode='trilinear',
        align_corners=False,
    )

    if selected_smooth in [3, 5]:
        disp_hr = apply_avg_pool3d(disp_hr, selected_smooth, num_repeats=3)
        
    return disp_hr
