# Pretrain anatomix

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
```

PRs are very welcome! There are some code snippets in the current repository that point out where to make these changes and I can help if someone gets started.

## Credits (and disclaimers)
This portion of the codebase is fork of this NeurIPS'22 [paper](https://github.com/mengweiren/longitudinal-representation-learning/tree/main)'s codebase, which itself a heavily modified fork of the [CUT](https://github.com/taesungp/contrastive-unpaired-translation/tree/master) repo.

As a result, this is more "research code" and fragile than the rest of the repository. Please open issues and/or pull requests if you spot any issues.