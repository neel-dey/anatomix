#!/usr/bin/env python

"""
Main training script for pretraining models.

This script handles the full training loop, including:
- Loading experiment options and datasets
- Model creation and initialization
- Training and validation loops
- Checkpointing and logging

Context:
- Uses options from pretraining/options/train_options.py
- Datasets are created via pretraining/data
- Models are created via pretraining/models
- Visualization and tensor saving via pretraining/util

"""

from copy import deepcopy
import os
import sys
import time
import torch
import random
import numpy as np
# Add parent directory to path for module imports
sys.path.append("../")

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualization import Visualizer
from util.util import save_tensor

import torch.backends.cudnn as cudnn

cudnn.benchmark = True  # our input sizes are always fixed

# -------------------------------
# Load experiment settings
# -------------------------------

opt = TrainOptions().parse()  # Parse command-line options

# Set random seeds for reproducibility
torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)

# -------------------------------
# Create training and validation datasets
# -------------------------------

# Training dataset
train_dataset = create_dataset(opt)
train_dataset_size = len(train_dataset)
bs = opt.batch_size
crop = opt.crop_size
resize = opt.resize
print(f"The number of training images = {train_dataset_size}")

# Validation dataset (deepcopy to avoid side effects)
val_opt = deepcopy(opt)
val_opt.isTrain = False
val_opt.batch_size = 1
val_opt.crop_size = -1  # Use full image for validation

val_dataset = create_dataset(val_opt)
val_dataset_size = len(val_dataset)
print("The number of validation images = %d" % val_dataset_size)
opt.isTrain = True
opt.batch_size = bs
opt.crop_size = crop

# -------------------------------
# Create model and visualizer
# -------------------------------

print(f"Model: {opt.model}")
model = create_model(opt)

# -------------------------------
# Training bookkeeping variables
# -------------------------------

total_iters = 0  # Total number of training iterations
optimize_time = 0.1

# Loss tracking setup
last_eval_loss = -1
best_evaluation_loss = 9999
last_train_loss = -1
cur_train_loss = 99

# -------------------------------
# Resume from checkpoint if requested
# -------------------------------
#
# We peek at latest_train_state.pth (small file) to recover the exact
# `total_iters` of the previous run, then set opt.epoch so the matching
# numbered network weights are loaded by model.setup() later. The full
# optimizer/scheduler/scaler restore happens after model.setup() via
# model.load_training_state().

if opt.continue_train:
    train_state_path = os.path.join(model.save_dir, "latest_train_state.pth")
    if os.path.exists(train_state_path):
        # CPU-only peek: don't materialize optimizer tensors on GPU yet.
        peek = torch.load(train_state_path, map_location="cpu")
        total_iters = int(peek["total_iters"])
        opt.epoch = str(total_iters)
        opt.epoch_count = total_iters // train_dataset_size
        print(
            f"Resuming from iter {total_iters} (epoch {opt.epoch_count}); "
            f"will load {opt.epoch}_net_*.pth and latest_train_state.pth"
        )
    else:
        print(
            f"continue_train requested but no {train_state_path} found; "
            "starting from scratch"
        )
        opt.continue_train = False
        opt.epoch = "0"
        opt.epoch_count = 0

# Defensive best-val read (older runs may have written this file even
# without a paired train_state; load_training_state will overwrite if
# present).
best_val_txt = os.path.join(model.save_dir, "best_val_loss.txt")
if opt.continue_train and os.path.exists(best_val_txt):
    with open(best_val_txt, "r") as f:
        best_evaluation_loss = float(f.readline().rstrip())
    print(f"Load best_val_loss.txt : {best_evaluation_loss:.6f}")

# Build the SummaryWriter AFTER we know total_iters so purge_step drops
# any prior events at or after the resume step. Without this, TensorBoard
# would overlay old and new scalar curves at duplicate step values.
print(f"* Creating tensorboard summary writer {model.save_dir}")
model.visualizer = Visualizer(opt, purge_step=total_iters if total_iters > 0 else None)

est_iters = (
    int(train_dataset_size / bs) * bs * (opt.n_epochs + opt.n_epochs_decay)
)
print(
    "Training start from epoch %d, total epochs: %d - est iters: %d"
    % ((opt.epoch_count, opt.n_epochs + opt.n_epochs_decay, est_iters))
)
print(f"Starting iters: {total_iters}")

# -------------------------------
# Main training loop
# -------------------------------

times = []
t_data = 0  # Time spent loading data

stop_now = False  # set by inner loop when max_iters is reached
for epoch in range(
    opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1
):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    if epoch > opt.stop_epoch:
        print(f"stop training at epoch {epoch}")
        break
    if stop_now:
        break

    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    # Set epoch for dataset (for shuffling, etc.)
    train_dataset.set_epoch(epoch)

    # -------------------------------
    # Iterate over training batches
    # -------------------------------
    for i, data in enumerate(train_dataset):
        batch_size = data["A"].size(0)
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += batch_size
        epoch_iter += batch_size

        # Synchronize CUDA for accurate timing
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        optimize_start_time = time.time()

        # Data-dependent initialization (first batch of first epoch)
        if epoch == opt.epoch_count and i == 0:
            print("Data dependent initialization")
            for k in set(data["keys"][0]):
                print(k, data[k].size())
            model.data_dependent_initialize(data)
            model.setup(opt)  # Load and print networks, create schedulers
            model.train()

            # Restore optimizer/scheduler/scaler state (only meaningful on
            # resume; returns None for cold starts or warm-start runs).
            # Note: total_iters was already set from the peek above; don't
            # restore it from extras here — total_iters has been incremented
            # once for this first iter, so re-overwriting would replay an iter.
            if opt.continue_train:
                extras = model.load_training_state()
                if extras is not None:
                    best_evaluation_loss = float(
                        extras.get("best_evaluation_loss", best_evaluation_loss)
                    )
                    last_eval_loss = float(
                        extras.get("last_eval_loss", last_eval_loss)
                    )
                    print(
                        f"Restored training state at iter "
                        f"{extras.get('total_iters')}; best_val="
                        f"{best_evaluation_loss:.6f}"
                    )

            # Compile:
            print('Compiling models...')
            model.netG = torch.compile(model.netG, mode="default")
            if hasattr(model, "netF"):
                model.netF = torch.compile(model.netF, mode="default")

        # Print info every print_freq iterations
        verbose = (total_iters % opt.print_freq == 0)
        if verbose:
            print(f"Start of iters [{total_iters}]")

        model.set_input(
            data, verbose
        )  # unpack data from dataset and apply preprocessing
        model.optimize_parameters(
            total_iters
        )  # calculate loss functions, get gradients, update network weights

        
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        del data
        if len(opt.gpu_ids) > 0:
            torch.cuda.empty_cache()

        optimize_time = (
            (time.time() - optimize_start_time) / batch_size * 0.005
            + 0.995 * optimize_time
        )

        # -------------------------------
        # Visualization and Logging
        # -------------------------------

        # Display current results on tensorboard
        if total_iters % opt.display_freq == 0:
            model.visualizer.display_current_results(
                model.get_current_visuals(), total_iters, board_name="train"
            )

        if (
            total_iters % opt.print_freq == 0
        ):  # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            model.visualizer.print_current_losses(
                total_iters, epoch, epoch_iter, losses, optimize_time, t_data
            )
            model.visualizer.plot_current_losses(total_iters, losses)
            model.visualizer.writer.add_scalar(
                "learning_rate",
                model.optimizers[0].param_groups[0]["lr"],
                total_iters,
            )

        if (
            total_iters % opt.save_latest_freq == 0
        ):  # cache our latest model every <save_latest_freq> iterations
            print(
                f"saving the latest model (epoch {epoch}, total_iters {total_iters})"
            )
            print(
                opt.name
            )  # it's useful to occasionally show the experiment name on console
            save_suffix = (
                f"iter_{total_iters}" if opt.save_by_iter else "latest"
            )
            model.save_networks(save_suffix)
            visuals = model.get_current_visuals()  # get image results
            if len(visuals) > 0:
                save_tensor(
                    "nii",
                    os.path.join(model.save_dir, "nii_latest/"),
                    "train",
                    visuals,
                )

        iter_data_time = time.time()  # Reset data loading timer

        # -------------------------------
        # Evaluation on Validation Set
        # -------------------------------

        if total_iters % opt.evaluation_freq == 0:
            model.save_networks(total_iters)
            model.save_training_state({
                "total_iters": total_iters,
                "epoch": epoch,
                "best_evaluation_loss": best_evaluation_loss,
                "last_eval_loss": last_eval_loss,
            })
            model.eval()

            cur_eval_loss = 0
            cnt_data = 0

            with torch.no_grad():
                print(f"{total_iters} iters, Evaluation starts ...")
                for vcnt, data in enumerate(
                    val_dataset
                ):  # inner loop within one epoch
                    if vcnt > opt.n_val_during_train:
                        break
                    model.set_input(
                        data, verbose=True
                    )  # unpack data from dataset and apply preprocessing
                    model.forward()  # calculate loss functions, get gradients, update network weights
                    cur_eval_loss += model.compute_G_loss(-1).item()
                    cnt_data += 1

                del data
                if len(opt.gpu_ids) > 0:
                    torch.cuda.empty_cache()
                cur_eval_loss = cur_eval_loss / cnt_data

                print(
                    f"Best validation loss: {best_evaluation_loss}, current validation loss {cur_eval_loss}"
                )

                # Display validation results
                model.visualizer.display_current_results(
                    model.get_current_visuals(), total_iters, board_name="val"
                )

                # Plot validation loss
                delta = abs(last_eval_loss - cur_eval_loss)
                model.visualizer.plot_current_losses(
                    total_iters, {"current_val": cur_eval_loss}
                )

                # Save best model if validation loss improves
                if cur_eval_loss < best_evaluation_loss:
                    best_evaluation_loss = cur_eval_loss
                    print(
                        f"Saving model with best validation loss: {best_evaluation_loss}, last validation loss {last_eval_loss}, delta {delta}"
                    )
                    model.save_networks("best_val")
                    model.visualizer.plot_current_losses(
                        total_iters, {"best_val": best_evaluation_loss}
                    )
                    with open(os.path.join(model.save_dir, "best_val_loss.txt"), "w") as f:
                        f.writelines(f"{best_evaluation_loss}")
                last_eval_loss = cur_eval_loss

                # Plateau LR scheduler step
                if opt.lr_policy == "plateau":
                    model.update_learning_rate(cur_eval_loss)

            model.train()
        if verbose:
            print(f"End of iters [{total_iters}]")

        if opt.max_iters > 0 and total_iters >= opt.max_iters:
            print(
                f"Reached max_iters={opt.max_iters} at total_iters={total_iters}; "
                "saving final train_state and stopping."
            )
            model.save_networks(total_iters)
            model.save_training_state({
                "total_iters": total_iters,
                "epoch": epoch,
                "best_evaluation_loss": best_evaluation_loss,
                "last_eval_loss": last_eval_loss,
            })
            stop_now = True
            break

    # ----------------------------------
    # Cache model every <save_freq> its
    # ----------------------------------

    if total_iters % opt.save_freq == 0:
        print(
            "saving the model at the end of epoch %d, iters %d"
            % (epoch, total_iters)
        )
        model.save_networks("latest")
        model.save_networks(total_iters)
        model.save_training_state({
            "total_iters": total_iters,
            "epoch": epoch,
            "best_evaluation_loss": best_evaluation_loss,
            "last_eval_loss": last_eval_loss,
        })
        visuals = model.get_current_visuals()  # get image results
        if len(visuals) > 0:
            save_tensor(
                "nii",
                os.path.join(model.save_dir, "nii_latest/"),
                "train",
                visuals,
            )

    # Print epoch summary
    print(
        "Total iters %d, End of epoch %d / %d \t Time Taken: %d sec"
        % (
            total_iters,
            epoch,
            opt.n_epochs + opt.n_epochs_decay,
            time.time() - epoch_start_time,
        )
    )
    if not opt.lr_policy == "plateau":
        model.update_learning_rate()  # update learning rates at the end of every epoch.
