# anatomix

### [Paper](TBD)

`anatomix` is a general-purpose feature extractor for 3D volumes. For any new biomedical dataset or task,
- It's out-of-the-box features (shown below) are invariant to most forms of nuisance imaging variation.
- It's out-of-the-box weights are a good initialization for finetuning when given limited annotations.

This respectively leads to:
- SOTA 3D training-free multi-modality image registration
- SOTA 3D few-shot semantic segmentation

(all without any dataset or domain specific pretraining)

![Features produced by network](https://www.neeldey.com/files/highlight_collage_v2.png)

`anatomix` is just a pretrained UNet! Use it for whatever you like.

**How?** It's weights were contrastively pretrained on wildly variable synthetic 
volumes to learn approximate appearance invariance and pose 
equivariance for images with randomly sampled biomedical shape configurations with
random intensities and artifacts.

## Roadmap

The current repository is just an initial push. It will be refactored 
and some quality-of-life and core aspects will be pushed as well in the coming weeks.
These include:
- [ ] Contrastive pretraining code 
- [ ] Jupyter/Colab tutorials
- [ ] General-purpose registration interface with [ANTs](https://github.com/ANTsX/ANTs)
- [ ] Dataset-specific modeling details for paper

## Folder organization

This repository contains:
- Model weights for `anatomix` (and its variants)
- Scripts for 3D nonrigid registration using `anatomix` features
- Scripts for finetuning `anatomix` weights for semantic segmentation.

Each subfolder (described below) has its own README to detail its use.

```bash
root-directory/
│
├── model-weights/                          # Pre-trained model weights
│
├── anatomix/model/                         # Model definition and architecture
│
├── anatomix/synthetic-data-generation/     # Scripts to generate synthetic training data
│
├── anatomix/registration/                  # Scripts for registration using the pretrained model
│
├── anatomix/segmentation/                  # Scripts for fine-tuning the model for semantic segmentation
│
└── README.md                               # This file
```

## Install dependencies:

All scripts will require the `anatomix` environment defined below to run.

```
conda create -n anatomix python=3.9
conda activate anatomix
pip install numpy==1.24.1 nibabel scipy scikit-image nilearn h5py matplotlib
pip3 install torch torchvision torchaudio
pip install monai==1.3.2 tensorboard
```

## Acknowledgements

Portions of this repository have been taken from the [Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation) 
and [ConvexAdam](https://github.com/multimodallearning/convexAdam) repositories and modified. Thanks!

## License

We use the standard MIT license.