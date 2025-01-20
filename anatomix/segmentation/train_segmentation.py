import logging
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.visualize import plot_2d_or_3d_image

from anatomix.segmentation.segmentation_utils import (
    load_model,
    save_ckp,
    worker_init_fn,
    get_train_transforms,
    get_val_transforms,
    data_handler,
)

torch.multiprocessing.set_sharing_strategy('file_system')


def main(opt):
    os.makedirs(
        'finetuning_runs/checkpoints/{}'.format(opt.exp_name),
        exist_ok=True,
    )
    os.makedirs(
        'finetuning_runs/runs/{}/'.format(opt.exp_name),
        exist_ok=True,
    )

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    trimages, trsegs, vaimages, vasegs = data_handler(
        opt.dataset, opt.train_amount, opt.n_iters_per_epoch, opt.batch_size,
    )

    print('Training cache: {} images {} segs'.format(len(trimages), len(trsegs)))
    print('Validation set: {} images {} segs'.format(len(vaimages), len(vasegs)))

    train_files = [
        {"image": img, "label": seg} for img, seg in zip(trimages, trsegs)
    ]
    val_files = [
        {"image": img, "label": seg} for img, seg in zip(vaimages, vasegs)
    ]
    
    # define transforms for image and segmentation
    train_transforms = get_train_transforms(opt.crop_size)
    val_transforms = get_val_transforms()

    # create a training data loader
    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=8,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1, 
        num_workers=0,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn,
        shuffle=True,
    )

    post_trans_pred = Compose(
        [Activations(softmax=True, dim=1), AsDiscrete(argmax=True, dim=1)]
    )

    # Create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    new_model = load_model(
        opt.pretrained_ckpt,
        opt.n_classes,
        device,
    )
    
    # Create Dice + CE loss function
    loss_function = monai.losses.DiceCELoss(
        softmax=True, to_onehot_y=True, include_background=False,
    )
    # Track Dice loss for validation
    valloss_function = monai.losses.DiceLoss(
        softmax=True, to_onehot_y=True, include_background=False,
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        new_model.parameters(), opt.lr, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.n_epochs
    )

    # start a typical PyTorch training
    val_interval = opt.val_interval
    best_val_loss = 10000000000
    epoch_loss_values = list()
    writer = SummaryWriter(
        log_dir='finetuning_runs/runs/{}/'.format(opt.exp_name),
        comment='_segmentor',
    )

    # Training loop
    for epoch in range(opt.n_epochs):
        print("-" * 10)
        print("epoch {:04d}/{:04d}".format(epoch + 1, opt.n_epochs))
        new_model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device) 
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            
            outputs = new_model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar(
                "train_loss", loss.item(), epoch_len * epoch + step,
            )

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        scheduler.step()

        # Plotting:
        with torch.no_grad():
            if (epoch + 1) % val_interval == 0:
                print('got to image plotter')
                plot_2d_or_3d_image(
                    inputs, epoch + 1, writer, index=0, tag="train/image",
                )
                plot_2d_or_3d_image(
                    labels/(opt.n_classes + 1.),
                    epoch + 1,
                    writer,
                    index=0,
                    tag="train/label",
                )
                plot_2d_or_3d_image(
                    post_trans_pred(outputs)/(opt.n_classes + 1.),
                    epoch + 1,
                    writer,
                    index=0,
                    tag="train/output",
                )
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation and checkpointing loop:
        if (epoch + 1) % val_interval == 0:
            new_model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                val_loss = 0.0
                valstep = 0
                for val_data in val_loader:
                    val_images = val_data["image"].to(device) 
                    val_labels = val_data["label"].to(device)
                    roi_size = (opt.crop_size, opt.crop_size, opt.crop_size)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size,
                        new_model, overlap=0.7,
                    )
                    val_loss += valloss_function(val_outputs, val_labels)
                    valstep += 1
                val_loss = val_loss / valstep

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss_epoch = epoch + 1
                    torch.save(
                        new_model.state_dict(),
                        "finetuning_runs/checkpoints/{}/"
                        "best_dict_epoch{:04d}.pth".format(
                            opt.exp_name, epoch + 1,
                        ),
                    )
                    print("saved new best loss model")

                print(
                    "current epoch: {} current mean dice: {:.4f}"
                    " best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, val_loss.item(),
                        best_val_loss.item(), best_loss_epoch,
                    )
                )
                writer.add_scalar(
                    "val_loss_mean_dice", val_loss.item(), epoch + 1
                )
                # plot the last model output as GIF image in TensorBoard 
                # with the corresponding image and label
                plot_2d_or_3d_image(
                    val_images, epoch + 1, writer, index=0, tag="Val/image",
                )
                plot_2d_or_3d_image(
                    val_labels/(opt.n_classes + 1.),
                    epoch + 1,
                    writer,
                    index=0,
                    tag="Val/label"
                )
                plot_2d_or_3d_image(
                    post_trans_pred(val_outputs)/(opt.n_classes + 1.),
                    epoch + 1,
                    writer,
                    index=0,
                    tag="Val/output",
                )

        if (epoch + 1) % val_interval == 0:
            checkpoint = {
                "state_dict": new_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_ckp(
                checkpoint,
                'finetuning_runs/checkpoints/{}/epoch{:04d}.pth'.format(
                    opt.exp_name, epoch+1
                ),
            )
                
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--dataset', type=str, default='./dataset/',
        help="Directory where image and label *.nii.gz files are stored.",
    )
    parser.add_argument(
        '--n_epochs', type=int, default=500,
        help="Number of epochs. "
        "An epoch is defined as n_iters_per_epoch training batches",
    )
    parser.add_argument(
        '--n_iters_per_epoch', type=int, default=75,
        help="Number of training batches per epoch",
    )
    parser.add_argument(
        '--n_classes', type=int, default=4,
        help="Number of classes to segment. Does not include background class",
    )
    parser.add_argument(
        '--val_interval', type=int, default=2,
        help="Do a valid. and checkpointing loop every val_interval epochs",
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4,
        help="Adam step size",
    )
    parser.add_argument(
        '--crop_size', type=int, default=128,
        help="Crop size to train on",
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help="Batch size to train with",
    )
    parser.add_argument(
        '--train_amount', type=int, default=3,
        help="No. of training samples to use for few-shot training",
    )
    parser.add_argument(
        '--pretrained_ckpt',
        type=str,
        default='../../model-weights/anatomix.pth',
        help="Default points to model weights path. "
        "Set to 'scratch' for random initialization",
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='demo',
        help="Prefix to attach to training logs in folder and file names",
    )

    args = parser.parse_args()

    main(args)
