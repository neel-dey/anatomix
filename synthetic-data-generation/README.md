# Synthetic Data Generation Overview

![Data generation overview](https://www.neeldey.com/files/data_generation_v2.png)

This folder contains scripts to download the base TotalSegmentator label files,
preprocess them, generate 3D label ensembles, and then synthesize paired 3D
volume pairs for contrastive pretraining.

Running this on a machine with several CPU cores would be ideal as you can
increase the `--max_workers` flag.

## One-call run through

To run everything in one go: `./generate_training_data.sh`.

## Step-by-step run through

### Get templates:

Download and unzip the TotalSegmentator dataset: 

```bash
wget https://zenodo.org/records/6802614/files/Totalsegmentator_dataset.zip
unzip Totalsegmentator_dataset.zip
```

Preprocess TotalSegmentator for our purposes with 
```bash
python step0_preprocess_totalsegmentator.py --totalsegmentator_path /path/to/totalsegmentator/Totalsegmentator_dataset/
```

Full CLI:
```bash
$ python step0_preprocess_totalsegmentator.py -h
usage: step0_preprocess_totalsegmentator.py [-h] [--totalsegmentator_path TOTALSEGMENTATOR_PATH] [--max_workers MAX_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --totalsegmentator_path TOTALSEGMENTATOR_PATH
                        Path to unzipped TotalSegmentator v1 dataset
  --max_workers MAX_WORKERS
                        Maximum number of worker processes to use (default: None, corresponding to all cores)
```

### Generate ten example 3D label ensembles:

This will generate ten example 3D label ensembles using the downloaded TotalSegmentator labels.

```bash
python step1_generate_labels.py --n_ensembles 10 --templatedir /path/to/totalsegmentator/Totalsegmentator_dataset/
```

Outputs will be saved in `./label_ensembles/` by default.

Full CLI:
```bash
$ python step1_generate_labels.py -h
usage: step1_generate_labels.py [-h] [--n_ensembles N_ENSEMBLES] [--min_templates MIN_TEMPLATES] [--max_templates MAX_TEMPLATES] [--side_length SIDE_LENGTH] [--templatedir TEMPLATEDIR]
                                [--savedir SAVEDIR] [--max_workers MAX_WORKERS]

Generate 3D label ensembles

optional arguments:
  -h, --help            show this help message and exit
  --n_ensembles N_ENSEMBLES
                        Number of 3D label ensemble volumes to generate
  --min_templates MIN_TEMPLATES
                        Minimum number of shapes to include in each ensemble
  --max_templates MAX_TEMPLATES
                        Maximum number of shapes to include in each ensemble
  --side_length SIDE_LENGTH
                        Side length of the generated volumes
  --templatedir TEMPLATEDIR
                        Path to unzipped and preprocessed TotalSegmentator data
  --savedir SAVEDIR     Directory to save the generated label ensembles
  --max_workers MAX_WORKERS
                        Maximum number of workers for parallel processing (default: None, corresponding to all cores)
```

### Generate ten example 3D synthetic volume pairs of contrastive views:

This script will use the ten example label ensembles generated above to generate 
paired volumes corresponding to two contrastive views:

```bash
python step2_generate_views.py --end_idx 10
```

Output views will be saved in `./synthesized_views/` by default.

```bash
$ python step2_generate_views.py -h
usage: step2_generate_views.py [-h] [--start_idx START_IDX] [--end_idx END_IDX] [--ensembledir ENSEMBLEDIR] [--savedir SAVEDIR] [--max_workers MAX_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --start_idx START_IDX
                        Starting index of list of label ensemble files to process
  --end_idx END_IDX     Ending index of list of label ensemble files to process
  --ensembledir ENSEMBLEDIR
                        Path to where the synthetic label ensembles are saved
  --savedir SAVEDIR     Path to save synthesized volumes to
  --max_workers MAX_WORKERS
                        Maximum number of worker processes to use

```
