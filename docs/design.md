# Design Document: Pre-trained Model of Multi-Modal Resting State

## Context and score

## Goals and non-goals

## Design

(trade-offs, choices)

## System-context-diagram

## APIs

## Data storages

## Psuedo code

## Alternatives considered

## Open questions

## Cross-cutting concerns

such as security, privacy, and observability

## Bill of materials

Here is a list of the software packages that will be installed on the machine. These are installed via the `environment.yml` conda file in the root of the repository.

### General
- Mamba (via Module)
- Python (v3.11)
- Git
- Git LFS
- NodeJS/NPM
- Jupyter Notebook, ipykernel
- flake8
- Seaborn, Matplotlib

### Data
- Boto3
- OpenNeuro CLI
- AWS CLI
- DVC
- SafeTensor
- XArray, h5netcdf
- PyYAML

### Neuroimaging
- MNE
- fMRIPrep
- Nilearn

### ML/DL
- PyTorch (v2)
- PyTorch Lightning (v2)
- TensorBoard
- torchmetrics
- scikit-learn

### DevOps
- Docker
- Singularity
