# Few-shot segmentation finetuning of anatomix weights

### [Colab tutorial on how to prepare data and finetune for 3D segmentation with anatomix and MONAI](https://colab.research.google.com/drive/1WBslSRLgAAMq6o5YFif1y0kaW9Ac15XK?usp=sharing)

Using only a few annotated volumes (3, 3, and 1 for the datasets in the screenshot below), 
we can finetune the pretrained anatomix weights to significantly improve segmentation
performance:

![Few-shot finetuning results](https://www.neeldey.com/files/qualitative-segmentation.png)

This subfolder contains a demo training script to finetune our pretrained network on
your own data for semantic segmentation in the few-shot regime.

The segmentation finetuning script is built based on MONAI.

## Data organization

Organize your training and validation data as below. The few-shot training script will
take an argument as to how many training volumes to do finetuning with.

```bash
dataset/
│
├── imagesTr/                         # Image niftis (*.nii.gz) for training set
│
├── labelsTr/                         # Label niftis (*.nii.gz) for training set
│
├── imagesVal/                        # Image niftis (*.nii.gz) for validation set
│
├── labelsVal/                        # Label niftis (*.nii.gz) for validation set
│
├── imagesTs/                         # Image niftis (*.nii.gz) for testing set
│
└── labelsTs/                         # Label niftis (*.nii.gz) for testing set
```

All results reported in the paper are calculated separately on held-out testing
data that is not used by any of the training or validation scripts. Validation
sets in the `train_segmentation.py` scripts are used for model selection.


## Usage

Once you have your data organized, start a finetuning run for `NCLASSES`-segmentation 
(e.g., if you have 3 organs and background, `NCLASSES=3`) with `NVOLS` annotated 
finetuning volumes from `imagesTr` and `labelsTr` with our pretrained weights as:
```bash
python train_segmentation.py 
--dataset ./dataset/ \
--n_classes NCLASSES \
--train_amount NVOLS \
--pretrained_ckpt ../../model-weights/anatomix.pth
```

To train from scratch / random initialization instead:
```bash
python train_segmentation.py \
--dataset ./dataset/ \
--n_classes NCLASSES \
--train_amount NVOLS \
--pretrained_ckpt scratch
```

Logs and checkpoints are saved in `finetuning_runs/checkpoints/` and `finetuning_runs/runs/`, respectively.

See full CLI below:
```
$ python train_segmentation.py -h
usage: train_segmentation.py [-h] [--dataset DATASET] [--n_epochs N_EPOCHS] [--n_iters_per_epoch N_ITERS_PER_EPOCH] [--n_classes N_CLASSES] [--val_interval VAL_INTERVAL] [--lr LR]
                             [--crop_size CROP_SIZE] [--batch_size BATCH_SIZE] [--train_amount TRAIN_AMOUNT] [--pretrained_ckpt PRETRAINED_CKPT] [--exp_name EXP_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Directory where image and label *.nii.gz files are stored.
  --n_epochs N_EPOCHS   Number of epochs. An epoch is defined as n_iters_per_epoch training batches
  --n_iters_per_epoch N_ITERS_PER_EPOCH
                        Number of training batches per epoch
  --n_classes N_CLASSES
                        Number of classes to segment. Does not include background class
  --val_interval VAL_INTERVAL
                        Do a valid. and checkpointing loop every val_interval epochs
  --lr LR               Adam step size
  --crop_size CROP_SIZE
                        Crop size to train on
  --batch_size BATCH_SIZE
                        Batch size to train with
  --train_amount TRAIN_AMOUNT
                        No. of training samples to use for few-shot training
  --pretrained_ckpt PRETRAINED_CKPT
                        Default points to model weights path. Set to 'scratch' for random initialization
  --exp_name EXP_NAME   Prefix to attach to training logs in folder and file names
```
