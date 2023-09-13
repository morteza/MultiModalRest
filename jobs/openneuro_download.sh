#!/bin/bash -l
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH -c 1
#SBATCH --time=0-02:00:00
#SBATCH -p batch

spack/bin/spack install py-datalad
spack/bin/spack load py-datalad
cd %PROJ_DIR/data/ && datalad install -r https://datasets.datalad.org/openneuro
