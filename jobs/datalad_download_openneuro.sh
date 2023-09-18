#!/bin/bash -l
#SBATCH --job-name=openneuro_download
#SBATCH --chdir=/shared/projects/TPDLMMMRSEF/
#SBATCH --output=/shared/projects/TPDLMMMRSEF/logs/%x_%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=big
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu

########################################################################
# Name:         openneuro_datalad.sh
# Description:  This script is used to download OpenNeuro dataset using Datalad
#               into the shared project folder.
# Requirements: spack (py-datalad), module (compiler/GCC/12.2.0)
########################################################################

# Install datalad
spack/bin/spack install py-datalad
spack/bin/spack load py-datalad

# Download dataset
DATALAD_DIR=/shared/projects/TPDLMMMRSEF/datasets/datalad/
mkdir -p $DATALAD_DIR

# NOTE: this is being executed in the shared project folder (see --chdir in the slurm headers)
cd $DATALAD_DIR && datalad install -r https://datasets.datalad.org/openneuro
