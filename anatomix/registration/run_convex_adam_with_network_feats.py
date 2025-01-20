import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
import nibabel as nib
import os
import random
from sklearn.metrics import f1_score

from anatomix.registration.convex_adam_utils import (
    extract_features,
    load_model,
)
from anatomix.registration.instance_optimization import (
    run_stage1_registration,
    run_instance_opt,
    merge_features,
)

import warnings
warnings.filterwarnings("ignore")


# coupled convex optimisation with adam instance optimisation
def convex_adam(
    ckpt_path,
    expname,
    lambda_weight,
    grid_sp,
    disp_hw,
    selected_niter,
    selected_smooth,
    grid_sp_adam=2,
    ic=True,
    result_path='./',
    fixed_image=None,
    moving_image=None,
    use_mask=False,
    fixed_mask=None,
    moving_mask=None,
    fixed_minclip=None,
    fixed_maxclip=None,
    moving_minclip=None,
    moving_maxclip=None,
    warp_seg=False,
    fixed_seg=None,
    moving_seg=None,
    downscale_feat_scalar=0.1,
):
    """
    This function first extracts anatomix and MIND features from the input
    images, concatenates them, and then performs coupled convex optimization
    with Adam instance optimization for image registration.

    Parameters
    ----------
    ckpt_path : str
        Path to the model checkpoint.
    expname : str
        Name of the experiment for logging and output file naming.
    lambda_weight : float
        Diffusion regularization weight during Adam instance optimization.
    grid_sp : int
        Grid spacing for the optimization grid.
    disp_hw : int
        Discretized search space half-width for optimization.
    selected_niter : int
        Number of iterations for Adam instance optimization.
    selected_smooth : int
        Degree of post-smoothing for output displacements (optional).
    grid_sp_adam : int, optional
        Grid spacing for Adam instance optimization (default is 2).
    ic : bool, optional
        If True, enable inverse consistency (default is True).
    result_path : str, optional
        Directory path to save output results (default is './').
    fixed_image : str, optional
        Path to the fixed image file (*.nii.gz format).
    moving_image : str, optional
        Path to the moving image file (*.nii.gz format).
    use_mask : bool, optional
        Whether to use masks for registration (default is False).
    fixed_mask : str, optional
        Path to the fixed image mask file (*.nii.gz format).
    moving_mask : str, optional
        Path to the moving image mask file (*.nii.gz format).
    fixed_minclip : float, optional
        Minimum intensity clipping value for the fixed image (optional).
    fixed_maxclip : float, optional
        Maximum intensity clipping value for the fixed image (optional).
    moving_minclip : float, optional
        Minimum intensity clipping value for the moving image (optional).
    moving_maxclip : float, optional
        Maximum intensity clipping value for the moving image (optional).
    warp_seg : bool, optional
        Whether to warp a segmentation label map (default is False).
    fixed_seg : str, optional
        Path to the fixed image segmentation file (*.nii.gz format).
    moving_seg : str, optional
        Path to the moving image segmentation file (*.nii.gz format).
    downscale_feat_scalar : float, optional
        Scalar to downscale network feature intensities (default is 0.1).

    Returns
    -------
    None
        The function saves the output files, including warped images,
        displacement fields, and optionally warped label maps, to the 
        specified result directory.
    """
    
    # load model:
    print('Loading model')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    model = load_model(ckpt_path)
    
    # load images:
    affine_mtx = nib.load(fixed_image).affine
    fixedim = nib.load(fixed_image).get_fdata()
    movingim = nib.load(moving_image).get_fdata()
    
    fixed_ch0 = torch.from_numpy(
        fixedim[np.newaxis, np.newaxis, ...],
    ).float().cuda()
    
    moving_ch0 = torch.from_numpy(
        movingim[np.newaxis, np.newaxis, ...],
    ).float().cuda()

    # Get a prefix with which to name saved files:
    fname = os.path.basename(moving_image)
    if fname.endswith(".nii.gz"):
        movsavename = fname[:-7]  # Remove '.nii.gz'
    else:
        movsavename = os.path.splitext(fname)[0]

    # Get anatomix features:
    print('Running network on input images')
    pred_fixed, pred_moving = extract_features(
        fixedim,
        movingim,
        model,
        fixed_minclip,
        fixed_maxclip,
        moving_minclip,
        moving_maxclip,
    )

    # Downscale feature intensities
    # otherwise network features and MIND features are in different scales
    pred_fixed = pred_fixed * downscale_feat_scalar
    pred_moving = pred_moving * downscale_feat_scalar

    # load masks if provided:
    if use_mask:
        mask_fixed = torch.from_numpy(
            nib.load(fixed_mask).get_fdata()
        ).float().cuda()
        mask_moving = torch.from_numpy(
            nib.load(moving_mask).get_fdata()
        ).float().cuda()
    else:
        mask_fixed = None
        mask_moving = None

    # Merge network features with MIND features:
    mind_fixed, mind_moving, pred_fixed, pred_moving = merge_features(
        use_mask,
        pred_fixed,
        pred_moving,
        mask_fixed,
        mask_moving,
        fixed_ch0,
        moving_ch0,
    )

    H, W, D = pred_fixed.shape[-3:]

    # track timing
    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():      
        features_fix, features_mov = pred_fixed, pred_moving
        features_fix_smooth = F.avg_pool3d(
            features_fix, grid_sp, stride=grid_sp,
        )
        features_mov_smooth = F.avg_pool3d(
            features_mov, grid_sp, stride=grid_sp,
        )
        n_ch = features_fix_smooth.shape[1]

    # Run coupled convex optimization:
    disp_hr = run_stage1_registration(
        features_fix_smooth,
        features_mov_smooth,
        disp_hw,
        grid_sp,
        (H, W, D),
        n_ch,
        ic,
    )

    # run Adam instance optimisation
    if selected_niter > 0:
        disp_hr = run_instance_opt(
            disp_hr,
            features_fix,
            features_mov,
            grid_sp_adam,
            lambda_weight,
            (H, W, D),
            selected_niter,
            selected_smooth,
            lr=1,
        )
        
    # Timing tracking:
    torch.cuda.synchronize()
    print('case time: ', time.time() - t0)

    # Warp the image with the estimated displacement:
    grid1 = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).cuda(),
        (1, 1, H, W, D), 
        align_corners=False
    )
    disp0 = disp_hr.cuda().float().permute(0, 2, 3, 4, 1)
    denom = torch.tensor([H - 1, W - 1, D - 1]).cuda().view(1, 1, 1, 1, 3)
    disp0 = disp0 / denom * 2
    disp0 = disp0.flip(4)
    
    moved = F.grid_sample(
        torch.from_numpy(movingim[None, None, ...]).float().cuda(),
        (grid1 + disp0).float(),
        align_corners=False,
        mode='bilinear',
    )
    
    if warp_seg:
        # Load segmentations:
        fixseg = nib.load(fixed_seg).get_fdata()
        movseg = nib.load(moving_seg).get_fdata()
        
        # Warp moving segmentation to fixed space:
        moved_seg = F.grid_sample(
            torch.from_numpy(movseg[None, None, ...]).float().cuda(), 
            (grid1 + disp0).float(), 
            align_corners=False, 
            mode='nearest',
        )
        
        # Save the warped label map:
        nib.save(
            nib.Nifti1Image(
                moved_seg.squeeze().cpu().numpy(), 
                affine_mtx,
            ), 
            os.path.join(
                result_path,
                'labels_moved_{}_g{}_hw{}_l{}_ga{}_ic{}_{}.nii.gz'.format(
                    movsavename, grid_sp, disp_hw, lambda_weight,
                    grid_sp_adam, ic, expname,
                )
            )
        )
        
        # Compute and report Dice:
        # TODO: should write this to a file when this script gets updated
        # to also have batch mode
        print(
            'Dice: {}'.format(
                f1_score(
                    fixseg.flatten(),
                    moved_seg.cpu().numpy().flatten(),
                    average='macro',
                    labels=np.unique(fixseg).astype(int).tolist()[1:],
                )
            )
        )
    
    # Save output displacements:
    nib.save(
        nib.Nifti1Image(
            disp_hr.permute(0, 2, 3, 4, 1).squeeze().cpu().numpy(),
            affine_mtx,
        ),
        os.path.join(
            result_path,
            'disp_{}_g{}_hw{}_l{}_ga{}_ic{}_{}.nii.gz'.format(
                movsavename, grid_sp, disp_hw, lambda_weight,
                grid_sp_adam, ic, expname,
            )
        )
    )

    # Save output moved volume:
    nib.save(
        nib.Nifti1Image(
            moved.squeeze().cpu().numpy(),
            affine_mtx,
        ),
        os.path.join(
            result_path,
            'moved_{}_g{}_hw{}_l{}_ga{}_ic{}_{}.nii.gz'.format(
                movsavename, grid_sp, disp_hw, lambda_weight,
                grid_sp_adam, ic, expname,
            )
        )
    )

    return


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Run ConvexAdam optimization with proposed network feats."
    )  
    parser.add_argument(
        "--fixed", type=str, required=True,
        help="Path to the fixed image *.nii.gz file (required)."
    )
    parser.add_argument(
        "--moving", type=str, required=True,
        help="Path to the moving image *.nii.gz file (required)."
    )
    parser.add_argument(
        "--exp_name", type=str, required=True,
        help="Experiment name for logging and output purposes (required)."
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True,
        help="Path to the checkpoint for loading the model (required)."
    )
    parser.add_argument(
        "--result_path", type=str, default='./',
        help="Directory to save the output results. Default current directory."
    )
    parser.add_argument(
        "--lambda_weight", type=float, default=0.75,
        help="Diffusion reg weight during Adam inst opt. Default is 0.75"
    )
    parser.add_argument(
        "--grid_sp", type=int, default=2,
        help="Grid spacing for the optimization grid. Default is 2."
    )
    parser.add_argument(
        "--disp_hw", type=int, default=1,
        help="Discretized search space half-width for MIND. Default 1."
    )
    parser.add_argument(
        '--selected_niter', type=int, default=80,
        help="Number of iterations for Adam instance opt. Default is 80."
    )
    parser.add_argument(
        '--selected_smooth', type=int, default=0,
        help="Post-processing used by the original repo. We dont use it."
    )
    parser.add_argument(
        '--grid_sp_adam', type=int, default=2,
        help="Grid spacing for Adam instance opt. Default 2."
    )
    parser.add_argument(
        '--no-ic',
        action='store_false',
        dest='ic',
        help='Disable inverse consistency.',
    )
    parser.add_argument(
        '--use_mask',
        action='store_true',
        help='Use a registration mask.'
    )
    parser.add_argument(
        '--path_mask_fixed', type=str, default=None,
        help="If using masks, provide a *.nii.gz file for the fixed img.",
    )
    parser.add_argument(
        '--path_mask_moving', type=str, default=None,
        help="If using masks, provide a *.nii.gz file for the moving img.",
    )
    
    parser.add_argument(
        '--fixed_minclip', type=float, default=None,
        help="If clipping, clip minimum intensity of fixed img to this val.",
    )
    parser.add_argument(
        '--fixed_maxclip', type=float, default=None,
        help="If clipping, clip maximum intensity of fixed img to this val.",
    )
    parser.add_argument(
        '--moving_minclip', type=float, default=None,
        help="If clipping, clip minimum intensity of moving img to this val.",
    )
    parser.add_argument(
        '--moving_maxclip', type=float, default=None,
        help="If clipping, clip maximum intensity of moving img to this val.",
    )
    parser.add_argument(
        '--warp_seg',
        action='store_true',
        help='Warp the provided moving label map with estimated deformation.'
    )
    parser.add_argument(
        '--path_seg_fixed', type=str, default=None,
        help="If warping labels, provide a *.nii.gz file for the fixed label.",
    )
    parser.add_argument(
        '--path_seg_moving', type=str, default=None,
        help="If warping labels, provide a *.nii.gz file for the moving label",
    )

    
    args = parser.parse_args()

    convex_adam(
        args.ckpt_path,
        args.exp_name,
        args.lambda_weight,
        args.grid_sp,
        args.disp_hw,
        args.selected_niter,
        args.selected_smooth,
        args.grid_sp_adam,
        args.ic,
        args.result_path,
        args.fixed,
        args.moving,
        args.use_mask,
        args.path_mask_fixed,
        args.path_mask_moving,
        args.fixed_minclip,
        args.fixed_maxclip,
        args.moving_minclip,
        args.moving_maxclip,
        args.warp_seg,
        args.path_seg_fixed,
        args.path_seg_moving,
    )
