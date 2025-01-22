# anatomix

[![Paper
shield](https://img.shields.io/badge/arXiv-2411.02372-red.svg)](https://arxiv.org/abs/2411.02372)
[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### [Paper](https://arxiv.org/abs/2411.02372) | [Training-free registration](https://github.com/neel-dey/anatomix/tree/main/anatomix/registration) | [Few-shot segmentation](https://github.com/neel-dey/anatomix/tree/main/anatomix/segmentation)
#### [Colab Tutorial: 3D Feature Extraction & 3D Multimodal Registration](https://colab.research.google.com/drive/1shivu4GtUoiDzDrE9RKD1RuEm3OqXJuD?usp=sharing)
#### [Colab Tutorial: Finetune anatomix for 3D Few-shot Segmentation with MONAI](https://colab.research.google.com/drive/1WBslSRLgAAMq6o5YFif1y0kaW9Ac15XK?usp=sharing)

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
model.load_state_dict(
    torch.load("./model-weights/anatomix.pth"),
    strict=True,
)
```

See how to use it on real data for feature extraction or registration in [this tutorial](https://colab.research.google.com/drive/1shivu4GtUoiDzDrE9RKD1RuEm3OqXJuD?usp=sharing). Or if you want to finetune for your own task, check out [this tutorial](https://colab.research.google.com/drive/1WBslSRLgAAMq6o5YFif1y0kaW9Ac15XK?usp=sharing) instead.

## Install dependencies:

All scripts will require the `anatomix` environment defined below to run.

```
conda create -n anatomix python=3.9
conda activate anatomix
git clone https://github.com/neel-dey/anatomix.git
cd anatomix
pip install -e .
```

Or install the dependencies manually:
```
conda create -n anatomix python=3.9
conda activate anatomix
pip install numpy==1.24.1 nibabel scipy scikit-image nilearn h5py matplotlib torch tensorboard tqdm
pip install monai==1.3.2
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

## Roadmap

The current repository is just an initial push. It will be refactored 
and some quality-of-life and core aspects will be pushed as well in the coming weeks.
These include:
- [ ] Contrastive pretraining code 
- [x] Colab 3D feature extraction tutorial
- [x] Colab 3D multimodality registration tutorial
- [x] Colab 3D few-shot finetuning tutorial
- [ ] General-purpose registration interface with [ANTs](https://github.com/ANTsX/ANTs)
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

## License

We use the standard MIT license.
