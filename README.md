# Multi-Head Brain Auto-Encoder

This project demonstrates multi-input multi-output pipeline to extract spatio-temporal embedding from brain time-series (including resting-state fMRI and EEG).

## Setup

```bash
mamba env create -f environment.yml
mamba activate multihead
```

## Data

You can use DVC to import OpenNeuro dataset. For example, to import [ds002843](https://openneuro.org/datasets/ds002843/):

```bash
dvc import https://github.com/OpenNeuroDatasets/ds002843.git / -o data/ds002843
```

The data will be downloaded to `data/ds002843/` directory.
