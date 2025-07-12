# Pretrain anatomix

https://www.neeldey.com/anatomix/static/videos/step3-lossmechanism.mp4

This folder contains the code and scripts necessary to implement the supervised contrastive loss using large-scale paired synthetic 3D biomedical volumes with a shared segmentation map. The goal of this pretraining is to learn general-purpose, robust 3D feature representations that are stable against nuisance imaging variations.

## Usage
At a high level,
- Run the scripts in the `../synthetic-data-generation/` folder. This will create two HDF5 files for training and validation data. Each entry of the HDF5 file will have two (paired) images alongside their segmentation label.
- To start pretraining, `cd scripts` and run `python pretrain_anatomix.py` script. This will start training on the data from step 1 and save checkpoints and tensorboard logs to a new `./checkpoints/` folder.

A minimal call that pretrains the anatomix model will look like:
```
cd scripts
python pretrain_anatomix.py --name demo --dataroot /path/to/generated/h5_files/ 
```

Using the default 120K generated contrastive pairs, this will run for 600K iterations. The generated `./checkpoints/pretrain/demo/` folder will contain a `best_val_net_G.pth` set of weights that you can load into anatomix as in the base repo.

Full CLI:
```
$ python pretrain_anatomix.py -h
Pretrain anatomix with configurable arguments.

optional arguments:
  -h, --help            show this help message and exit
  --ckpt_dir CKPT_DIR   models are saved here
  --dataroot DATAROOT   path to images
  --name NAME           name of the run you're launching
  --n_epochs N_EPOCHS   number of epochs with the initial learning rate
  --n_epochs_decay N_EPOCHS_DECAY
                        number of epochs to start linearly decaying learning
                        rate to zero
  --crop_size CROP_SIZE
                        crop size
  --batch_size BATCH_SIZE
                        input batch size in terms of number of contrastive
                        paired volumes
  --dataset_mode DATASET_MODE
                        chooses datasets
  --model MODEL         chooses which model to use. only `supcl` supported as
                        of now
  --nce_T NCE_T         NCE temperature
  --ndims NDIMS         network dimension: 2|3. only 3 supported as of now.
  --input_nc INPUT_NC   number of input image channels
  --output_nc OUTPUT_NC
                        number of network output feature channels
  --ngf NGF             number of filters in the first conv layer
  --netF NETF           specify the feature network type. we use a patch
                        sampling MLP
  --n_mlps N_MLPS       number of MLP layers in netF
  --num_threads NUM_THREADS
                        number of threads for loading data
  --lr LR               initial learning rate for adam
  --print_freq PRINT_FREQ
                        frequency of showing training results on console
  --display_ncols DISPLAY_NCOLS
                        if positive, display all images in a single web panel
                        with certain number of images per row.
  --display_slice DISPLAY_SLICE
                        the slice index to display if inputs are 3D volumes
  --display_freq DISPLAY_FREQ
                        frequency of showing training results on screen
  --save_latest_freq SAVE_LATEST_FREQ
                        frequency of saving the latest results
  --save_freq SAVE_FREQ
                        frequency of saving checkpoints at the end of
                        iterationss
  --evaluation_freq EVALUATION_FREQ
                        evaluation freq
  --load_mode LOAD_MODE
                        load entries randomly (train) or in order (test time)
  --num_patches NUM_PATCHES
                        number of patches
  --lr_policy LR_POLICY
                        specify learning rate policy to use
  --init_type INIT_TYPE
                        network initialization strategy
  --n_val_during_train N_VAL_DURING_TRAIN
                        number of batches to sample during a validation step
  --lambda_NCE LAMBDA_NCE
                        weight for NCE loss
  --netF_nc NETF_NC     number of neurons for netF, the patch sampling MLP
  --normG NORMG         instance/batch/no/layer norm for base network
  --netG NETG           specify base network architecture
  --grad_accum_iters GRAD_ACCUM_ITERS
                        Gradient accumulation iterations
  --continue_train CONTINUE_TRAIN
                        continue training: load the latest model (True/False)
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU
  --nce_layers NCE_LAYERS
                        comma-separated list of layers for NCE loss
  --nce_weights NCE_WEIGHTS
                        comma-separated list of weights for NCE loss layers
  --seed SEED           Seed for torch, np, and random packages
  --apply_same_inten_augment APPLY_SAME_INTEN_AUGMENT
                        Whether to perform the same intensity augmentation on
                        view 1 & 2 (True/False)

```

## TODOs:
- [ ] Support masking the regions from where to sample patches
- [ ] Support 2D training
- [ ] Support multi-GPU training
- [ ] Support auto mixed precision

PRs are very welcome! There are some code snippets in the current repository that point out where to make these changes and I can help if someone gets started.

## Credits (and disclaimers)
This portion of the codebase is fork of this NeurIPS'22 [paper](https://github.com/mengweiren/longitudinal-representation-learning/tree/main)'s codebase, which itself a heavily modified fork of the [CUT](https://github.com/taesungp/contrastive-unpaired-translation/tree/master) repo. 

As a result, this is more "research code" and fragile than the rest of the repository. Please open issues and/or pull requests if you spot any issues.