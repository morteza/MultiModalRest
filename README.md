# Pre-trained Multi-Modal Resting-State Brain Model

This project demonstrates multi-modal multi-head pipeline to extract spatiotemporal embedding from resting-state brain activities (including resting-state fMRI and EEG).

## Setup

```bash
mamba env create -f environment.yml
mamba activate MultiModalRest
```

## Data

Use DVC to import OpenNeuro datasets. For example, to import [ds002843](https://openneuro.org/datasets/ds002843/):

```bash
dvc import https://github.com/OpenNeuroDatasets/ds002843.git / -o data/ds002843
```

The data will be downloaded to `data/ds002843/` directory.
