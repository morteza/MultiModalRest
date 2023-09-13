#!/bin/bash -l
#SBATCH --job-name=openneuro_download
#SBATCH --output=/shared/projects/TPDLMMMRSEF/logs/%x_%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=big
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu


spack/bin/spack install py-datalad
spack/bin/spack load py-datalad
cd %PROJ_DIR/data/ && datalad install -r https://datasets.datalad.org/openneuro
