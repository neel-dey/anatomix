[![Paper
shield](https://img.shields.io/badge/arXiv-2411.02372-red.svg)](https://arxiv.org/abs/2411.02372)
[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<a href="https://colab.research.google.com/drive/1shivu4GtUoiDzDrE9RKD1RuEm3OqXJuD?usp=sharing"><img alt="Colab tutorial 1" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# anatomix (ICLR'25)

### [Project page](http://www.neeldey.com/anatomix/) | [Paper](https://arxiv.org/abs/2411.02372) | [Training-free registration](https://github.com/neel-dey/anatomix/tree/main/anatomix/registration) | [Few-shot segmentation](https://github.com/neel-dey/anatomix/tree/main/anatomix/segmentation)
#### [Colab Tutorial: 3D Feature Extraction & 3D Multimodal Registration](https://colab.research.google.com/drive/1shivu4GtUoiDzDrE9RKD1RuEm3OqXJuD?usp=sharing)
#### [Colab Tutorial: Finetune anatomix for 3D Few-shot Segmentation with MONAI](https://colab.research.google.com/drive/1WBslSRLgAAMq6o5YFif1y0kaW9Ac15XK?usp=sharing)

> **New 1**: Added experimental 94M-parameter U-Net (`anatomix-dev`) and 26M-parameter 3D ViT (`anatomix-dev-vit`) models, both of which are trained on more and better data using better pretraining losses. Make sure to voxelwise normalize their features to have either unit norm or have zero mean with unit standard deviation prior to feature visualization or use in out-of-the-box feature-based applications like registration.

> **New 2**: Interested in using anatomix features to enhance segmentation training regardless of the test domain and how much data you have? Check out [MaskGen](https://arxiv.org/abs/2604.02564), a simple 5-line modification to standard training that uses anatomix to get SOTA domain generalization.

![Highlight collage](https://www.neeldey.com/files/anatomix_github_highlight.png)

`anatomix` is a general-purpose feature extractor for 3D volumes. For any new biomedical dataset or task,
- Its out-of-the-box features are invariant to most forms of nuisance imaging variation.
- Its out-of-the-box weights are a good initialization for finetuning when given limited annotations.

Without any dataset or domain specific pretraining, this respectively leads to:
- SOTA 3D training-free multi-modality image registration
- SOTA 3D few-shot semantic segmentation

**How?** It's weights were contrastively pretrained on wildly variable synthetic
volumes to learn approximate appearance invariance and pose
equivariance for images with randomly sampled biomedical shape configurations with
random intensities and artifacts.

## Load weights for inference / feature extraction / finetuning

`anatomix` is just a pretrained UNet! Use it for whatever you like.

Pull the model and weights from HuggingFace Hub:

```python
from anatomix.model.load_from_hf import load_from_hf

model = load_from_hf("anatomix")          # 6M params. ICLR2025 version of the pretrained model
```

Or use one of the **NEW** experimental models and weights available here:
```python
model = load_from_hf("anatomix-dev")        # 94M-parameter experimental UNet. More parameters, better features. Give it a shot.
# model = load_from_hf("anatomix-dev-vit")  # 26M-parameter experimental 3D ViT. Requires 128^3 inputs, but amenable to sliding window.
```
(**NOTE:** When visualizing features or when using these dev models for out-of-the-box applications like registration, make sure to voxelwise normalize the features across channels to have unit norm or zero mean with unit standard deviation, either will work.)

Or just load a local checkpoint:

```python
import torch
from anatomix.model.network import Unet

model = Unet(
    dimension=3,  # Only 3D supported for now
    input_nc=1,  # number of input channels
    output_nc=16,  # number of output channels
    num_downs=4,  # number of downsampling layers
    ngf=16,  # channel multiplier
)

# model = torch.compile(model)

model.load_state_dict(
    torch.load("./model-weights/anatomix.pth"),
    strict=True,
)
```

See how to use it on real data for feature extraction or registration in [this tutorial](https://colab.research.google.com/drive/1shivu4GtUoiDzDrE9RKD1RuEm3OqXJuD?usp=sharing). Or if you want to finetune for your own task, check out [this tutorial](https://colab.research.google.com/drive/1WBslSRLgAAMq6o5YFif1y0kaW9Ac15XK?usp=sharing) instead.

(If your task involves brains, abdomens, and generally interesting shapes, you might benefit from the two `anatomix-dev` models.)

## Install dependencies:

All scripts will require the `anatomix` environment defined below to run.

```
conda create -n anatomix python=3.11
conda activate anatomix
git clone https://github.com/neel-dey/anatomix.git
cd anatomix
./install.sh   # auto-detects CUDA: cu126 default, cu130 for CUDA-13 drivers, cpu if no GPU
```

Or install the dependencies manually:
```
conda create -n anatomix python=3.11
conda activate anatomix
# change the cuda wheel to `cu130` if you're on CUDA13 or `cpu` if you're doing cpu-only
pip install numpy nibabel scipy scikit-image nilearn h5py matplotlib tensorboard tqdm monai torchio SimpleITK natsort huggingface_hub dynamic-network-architectures torch==2.13.0 torchvision==0.28.0 --extra-index-url https://download.pytorch.org/whl/cu126
```

## Folder organization

This repository contains:
- Model weights for `anatomix` (and its variants)
- Scripts for 3D nonrigid registration using `anatomix` features
- Scripts for finetuning `anatomix` weights for semantic segmentation.

Each subfolder (described below) has its own README to detail its use.

```bash
root-directory/
│
├── model-weights/                          # Pretrained model weights
│
├── pretraining/                            # Pretraining code and scripts 
│
├── synthetic-data-generation/              # Scripts to generate synthetic training data
│
├── anatomix/model/                         # Model definition and architecture
│
├── anatomix/registration/                  # FireANTs-based 3D registration on anatomix features (anatomix-register.py)
│
├── anatomix/segmentation/                  # Scripts for fine-tuning the model for semantic segmentation
│
└── README.md                               # This file
```

## Roadmap

The current repository is just an initial push. It will be refactored 
and some quality-of-life and core aspects will be pushed as well in the coming weeks.
These include:
- [x] Contrastive pretraining code 
- [x] Colab 3D feature extraction tutorial
- [x] Colab 3D multimodality registration tutorial
- [x] Colab 3D few-shot finetuning tutorial
- [x] General-purpose registration interface with [FireANTs](https://github.com/rohitrango/FireANTs) — FireANTs is now the general registration backend via [`anatomix/registration/`](anatomix/registration/) (`anatomix-register.py`); see the [FireANTs](anatomix/registration/tutorials/anatomix_registration_fireants.ipynb) and (legacy ICLR'25) [ConvexAdam](anatomix/registration/tutorials/anatomix_registration_convexadam.ipynb) tutorials.
- [ ] Dataset-specific modeling details for paper


## Citation

If you find our work useful, please consider citing:

```
@misc{dey2024learninggeneralpurposebiomedicalvolume,
      title={Learning General-Purpose Biomedical Volume Representations using Randomized Synthesis}, 
      author={Neel Dey and Benjamin Billot and Hallee E. Wong and Clinton J. Wang and Mengwei Ren and P. Ellen Grant and Adrian V. Dalca and Polina Golland},
      year={2024},
      eprint={2411.02372},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02372}, 
}
```

## Acknowledgements

Portions of this repository have been taken from the [Contrastive Unpaired Translation](https://github.com/taesungp/contrastive-unpaired-translation) 
and [ConvexAdam](https://github.com/multimodallearning/convexAdam) repositories and modified. Thanks!
