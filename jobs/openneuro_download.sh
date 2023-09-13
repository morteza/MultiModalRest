#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --time=0-12:00:00
#SBATCH -p big

spack/bin/spack install py-datalad
spack/bin/spack load py-datalad
cd %PROJ_DIR/data/ && datalad install -r https://datasets.datalad.org/openneuro
