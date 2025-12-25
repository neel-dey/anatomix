import argparse
import subprocess
import sys
import shlex


def main(args):
    cmd = [
        sys.executable, "../trainers/train.py",
        "--checkpoints_dir", args.ckpt_dir,
        "--name", args.name,
        "--dataroot", args.dataroot,
        "--dataset_mode", args.dataset_mode,
        "--model", args.model,
        "--nce_T", str(args.nce_T),
        "--ndims", str(args.ndims),
        "--input_nc", str(args.input_nc),
        "--output_nc", str(args.output_nc),
        "--ngf", str(args.ngf),
        "--num_downs", str(args.num_downs),
        "--netF", args.netF,
        "--n_mlps", str(args.n_mlps),
        "--num_threads", str(args.num_threads),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--print_freq", str(args.print_freq),
        "--display_ncols", str(args.display_ncols),
        "--display_slice", str(args.display_slice),
        "--display_freq", str(args.display_freq),
        "--save_latest_freq", str(args.save_latest_freq),
        "--save_freq", str(args.save_freq),
        "--evaluation_freq", str(args.evaluation_freq),
        "--load_mode", args.load_mode,
        "--num_patches", str(args.num_patches),
        "--crop_size", str(args.crop_size),
        "--batch_size", str(args.batch_size),
        "--lr_policy", args.lr_policy,
        "--init_type", args.init_type,
        "--n_val_during_train", str(args.n_val_during_train),
        "--n_epochs", str(args.n_epochs),
        "--n_epochs_decay", str(args.n_epochs_decay),
        "--lambda_NCE", str(args.lambda_NCE),
        "--netF_nc", str(args.netF_nc),
        "--normG", args.normG,
        "--normF", args.normF,
        "--netG", args.netG,
        "--grad_accum_iters", str(args.grad_accum_iters),
        "--continue_train", str(args.continue_train),
        "--gpu_ids", str(args.gpu_ids),
        "--nce_layers", args.nce_layers,
        "--nce_weights", args.nce_weights,
        "--seed", str(args.seed),
        "--apply_same_inten_augment", str(args.apply_same_inten_augment)
    ]
    print("Running command:\n" + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain anatomix with configurable arguments."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="../checkpoints/pretrain/",
        help="models are saved here",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="../../synthetic-data-generation/h5_w_segs/",
        help="path to images",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="demo",
        help="name of the run you're launching",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=0,
        help="number of epochs with the initial learning rate",
    )
    parser.add_argument(
        "--n_epochs_decay",
        type=int,
        default=4,
        help="number of epochs to start linearly decaying learning rate to zero",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=128,
        help="crop size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="input batch size in terms of number of contrastive paired volumes",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="h5supcl",
        help="chooses datasets",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="supcl",
        help="chooses which model to use. only `supcl` supported as of now",
    )
    parser.add_argument(
        "--nce_T",
        type=float,
        default=0.33,
        help="NCE temperature",
    )
    parser.add_argument(
        "--ndims",
        type=int,
        default=3,
        help="network dimension: 2|3. only 3 supported as of now.",
    )
    parser.add_argument(
        "--input_nc",
        type=int,
        default=1,
        help="number of input image channels",
    )
    parser.add_argument(
        "--output_nc",
        type=int,
        default=16,
        help="number of network output feature channels",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=16,
        help="number of filters in the first conv layer",
    )
    parser.add_argument(
        "--num_downs",
        type=int,
        default=4,
        help="number of downsamples in encoder",
    )
    parser.add_argument(
        "--netF",
        type=str,
        default="mlp_sample",
        help="specify the feature network type. we use a patch sampling MLP",
    )
    parser.add_argument(
        "--n_mlps",
        type=int,
        default=3,
        help="number of MLP layers in netF",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="number of threads for loading data",
        )  # TODO: RAM memory leak if >0, investigate dataset class for fix
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="initial learning rate for adam",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="weight decay for adamw",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="frequency of showing training results on console",
    )
    parser.add_argument(
        "--display_ncols",
        type=int,
        default=2,
        help="if positive, display all images in a single web panel with certain number of images per row.",
    )
    parser.add_argument(
        "--display_slice",
        type=int,
        default=64,
        help="the slice index to display if inputs are 3D volumes",
    )
    parser.add_argument(
        "--display_freq",
        type=int,
        default=100,
        help="frequency of showing training results on screen",
    )
    parser.add_argument(
        "--save_latest_freq",
        type=int,
        default=400,
        help="frequency of saving the latest results",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=4000,
        help="frequency of saving checkpoints at the end of iterationss",
    )
    parser.add_argument(
        "--evaluation_freq",
        type=int,
        default=200,
        help="evaluation freq",
    )
    parser.add_argument(
        "--load_mode",
        type=str,
        default="twoview",
        help="load entries randomly (train) or in order (test time)",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=512,
        help="number of patches",
    )
    parser.add_argument(
        "--lr_policy",
        type=str,
        default="const_linear",
        help="specify learning rate policy to use",
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="kaiming",
        help="network initialization strategy",
    )
    parser.add_argument(
        "--n_val_during_train",
        type=int,
        default=50,
        help="number of batches to sample during a validation step",
    )
    parser.add_argument(
        "--lambda_NCE",
        type=float,
        default=1,
        help="weight for NCE loss",
    )
    parser.add_argument(
        "--netF_nc",
        type=int,
        default=256,
        help="number of neurons for netF, the patch sampling MLP",
    )
    parser.add_argument(
        "--normG",
        type=str,
        default="batch",
        help="instance/batch/no/layer norm for base network",
    )
    parser.add_argument(
        "--normF",
        type=str,
        default="batch",
        help="instance/batch/no/layer norm for MLP",
    )
    parser.add_argument(
        "--netG",
        type=str,
        default="unet",
        help="specify base network architecture",
    )
    parser.add_argument(
        "--grad_accum_iters",
        type=int,
        default=1,
        help="Gradient accumulation iterations",
    )
    parser.add_argument(
        "--continue_train",
        type=str,
        default="False",
        help="continue training: load the latest model (True/False)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        default=0,
        help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
    )
    parser.add_argument(
        "--nce_layers",
        type=str,
        default="27,31,38,45,52,65",
        help="comma-separated list of layers for NCE loss",
    )
    parser.add_argument(
        "--nce_weights",
        type=str,
        default="1,1,1,1,1,1",
        help="comma-separated list of weights for NCE loss layers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234567,
        help="Seed for torch, np, and random packages",
    )
    parser.add_argument(
        "--apply_same_inten_augment",
        type=str,
        default="False",
        help="Whether to perform the same intensity augmentation on view 1 & 2 (True/False)",
    )
    args = parser.parse_args()
    main(args)
