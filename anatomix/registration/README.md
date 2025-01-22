# Multi-modality registration with anatomix features

### [Colab tutorial to reproduce our Learn2Reg-AbdomenMRCT results](https://colab.research.google.com/drive/1shivu4GtUoiDzDrE9RKD1RuEm3OqXJuD?usp=sharing)

![Qualitative registration collage](https://www.neeldey.com/files/qualitative-registration-v2.png)

The demo script in this folder extracts pretrained network features from input
volumes and runs a registration solver on the features to align volumes
across imaging modalities.

This subfolder is a heavily modified fork of the [ConvexAdam repository](https://github.com/multimodallearning/convexAdam).

**Paper-specific note**: For the experiments in the paper, we use ROI masks and 
clip the CT values to [-450, 450] HU and register MRI volumes to CT volumes.


## Usage

Example usage to register `moving.nii.gz` to `fixed.nii.gz` (with optional
registration masks `moving_mask.nii.gz` and `fixed.nii.gz`), assuming that 
fixed is a CT volume:
```bash
python run_convex_adam_with_network_feats.py \
    --fixed fixed.nii.gz \
    --moving moving.nii.gz \
    --ckpt_path ../../model-weights/anatomix.pth \
    --exp_name demo \
    --use_mask \
    --path_mask_fixed fixed_mask.nii.gz \
    --path_mask_moving moving_mask.nii.gz \
    --fixed_minclip -450 \
    --fixed_maxclip 450
```

Entire CLI:
```bash
$ python run_convex_adam_with_network_feats.py -h

usage: run_convex_adam_with_network_feats.py [-h] --fixed FIXED --moving MOVING --exp_name EXP_NAME --ckpt_path CKPT_PATH
                                             [--result_path RESULT_PATH] [--lambda_weight LAMBDA_WEIGHT]
                                             [--grid_sp GRID_SP] [--disp_hw DISP_HW] [--selected_niter SELECTED_NITER]
                                             [--selected_smooth SELECTED_SMOOTH] [--grid_sp_adam GRID_SP_ADAM] [--no-ic]
                                             [--use_mask] [--path_mask_fixed PATH_MASK_FIXED]
                                             [--path_mask_moving PATH_MASK_MOVING] [--fixed_minclip FIXED_MINCLIP]
                                             [--fixed_maxclip FIXED_MAXCLIP] [--moving_minclip MOVING_MINCLIP]
                                             [--moving_maxclip MOVING_MAXCLIP] [--warp_seg]
                                             [--path_seg_fixed PATH_SEG_FIXED] [--path_seg_moving PATH_SEG_MOVING]

Run ConvexAdam optimization with proposed network feats.

options:
  -h, --help            show this help message and exit
  --fixed FIXED         Path to the fixed image *.nii.gz file (required).
  --moving MOVING       Path to the moving image *.nii.gz file (required).
  --exp_name EXP_NAME   Experiment name for logging and output purposes (required).
  --ckpt_path CKPT_PATH
                        Path to the checkpoint for loading the model (required).
  --result_path RESULT_PATH
                        Directory to save the output results. Default current directory.
  --lambda_weight LAMBDA_WEIGHT
                        Diffusion reg weight during Adam inst opt. Default is 0.75
  --grid_sp GRID_SP     Grid spacing for the optimization grid. Default is 2.
  --disp_hw DISP_HW     Discretized search space width for MIND. Default 1.
  --selected_niter SELECTED_NITER
                        Number of iterations for Adam instance opt. Default is 80.
  --selected_smooth SELECTED_SMOOTH
                        Post-processing used by the original repo. We dont use it.
  --grid_sp_adam GRID_SP_ADAM
                        Grid spacing for Adam instance opt. Default 2.
  --no-ic               Disable inverse consistency.
  --use_mask            Use a registration mask.
  --path_mask_fixed PATH_MASK_FIXED
                        If using masks, provide a *.nii.gz file for the fixed img.
  --path_mask_moving PATH_MASK_MOVING
                        If using masks, provide a *.nii.gz file for the moving img.
  --fixed_minclip FIXED_MINCLIP
                        If clipping, clip minimum intensity of fixed img to this val.
  --fixed_maxclip FIXED_MAXCLIP
                        If clipping, clip maximum intensity of fixed img to this val.
  --moving_minclip MOVING_MINCLIP
                        If clipping, clip minimum intensity of moving img to this val.
  --moving_maxclip MOVING_MAXCLIP
                        If clipping, clip maximum intensity of moving img to this val.
  --warp_seg            Warp the provided moving label map with estimated deformation.
  --path_seg_fixed PATH_SEG_FIXED
                        If warping labels, provide a *.nii.gz file for the fixed label.
  --path_seg_moving PATH_SEG_MOVING
                        If warping labels, provide a *.nii.gz file for the moving label
```
