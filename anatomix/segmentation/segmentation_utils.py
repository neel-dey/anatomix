import torch
import os
import numpy as np
from monai.networks.blocks import UnetOutBlock
from glob import glob

from monai.transforms import (
    ScaleIntensityd,
    Compose,
    LoadImaged,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandSpatialCropd,
    RandAffined,
    EnsureTyped,
    EnsureChannelFirstd,
)

# -----------------------------------------------------------------------------
# Loading pretrained model

import sys
sys.path.append('../model/')

import warnings
warnings.filterwarnings("ignore")

from network import Unet

def load_model(pretrained_ckpt, n_classes, device):
    model = Unet(3, 1, 16, 4, ngf=16).to(device)
    if pretrained_ckpt == 'scratch':
        print("Training from random initialization.")
        pass
    else:
        print("Transferring from proposed pretrained network.")
        model.load_state_dict(torch.load(pretrained_ckpt))
    fin_layer = UnetOutBlock(3, 16, n_classes + 1, False).to(device)
    new_model = torch.nn.Sequential(model, fin_layer)
    new_model.to(device)
    
    return new_model


# -----------------------------------------------------------------------------
# Misc. utilities

def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(
            worker_info.seed % (2 ** 32)
        )
    except AttributeError:
        pass


# -----------------------------------------------------------------------------
# augmentation definitions

def get_train_transforms(crop_size):
    """
    Get training data transforms based on the specified dataset.

    This function returns a composition of data transformation 
    functions for training a model. These are just base augmentations.
    For actual augmentations per dataset, refer to App. B of the submission.
    This will be made dataset-specific for public release.

    Parameters
    ----------
    crop_size : int
        The size of the crop to be applied to the images.

    Returns
    -------
    train_transforms : Compose
        A composed transform object containing the specified 
        transformations for the training dataset.
    """
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[crop_size, crop_size, crop_size],
                random_size=False,
            ),
            RandGaussianNoised(keys=["image"], prob=0.33),
            RandBiasFieldd(
                keys=["image"], prob=0.33, coeff_range=(0.0, 0.05)
            ),
            RandGibbsNoised(keys=["image"], prob=0.33, alpha=(0.0, 0.33)),
            RandAdjustContrastd(keys=["image"], prob=0.33),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.33,
                sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
            ),
            RandGaussianSharpend(keys=["image"], prob=0.33),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.2, 0.2, 0.2),
                shear_range=(0.2, 0.2, 0.2),
                spatial_size=(crop_size, crop_size, crop_size),
                padding_mode='zeros',
            ),
            ScaleIntensityd(keys="image"),
        ]
    )
    return train_transforms


def get_val_transforms():
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
        ]
    )
    return val_transforms


# -----------------------------------------------------------------------------
# Dataset handling


def data_handler(basedir, finetuning_amount=3, repeats=100, seed=12345):
    # Training set:
    trimages = sorted(
        glob(
            os.path.join(basedir, './imagesTr/*.nii.gz'),
        )
    )
    trsegs = sorted(
        glob(
            os.path.join(basedir, './labelsTr/*.nii.gz'),
        )
    )
    assert len(trimages) > 0
    assert len(trimages) == len(trsegs)
    
    # For few-shot seg, we randomly select a `finetuning_amount` of images
    trimages = np.random.RandomState(seed=seed).permutation(trimages).tolist()
    trsegs = np.random.RandomState(seed=seed).permutation(trsegs).tolist()
    trimages = trimages[:finetuning_amount]
    trsegs = trsegs[:finetuning_amount]

    # TODO: make logic better
    # Few-shot finetuning does not really have an idea of an "epoch", so we
    # pick a number (75 training iterations) as an epoch. To do that with a
    # batch size of 4 in all our finetuning experiments:
    trimages = trimages * repeats
    trsegs = trsegs * repeats

    # Get validation set niftis as normal:
    vaimages = sorted(
        glob(
            os.path.join(basedir, './imagesVal/*.nii.gz'),
        )
    )
    vasegs = sorted(
        glob(
            os.path.join(basedir, './labelsVal/*.nii.gz'),
        )
    )
    
    return trimages, trsegs, vaimages, vasegs