#!/bin/bash -l
#SBATCH --job-name=openneuro_s3
#SBATCH --chdir=/shared/projects/TPDLMMMRSEF/
#SBATCH --output=/shared/projects/TPDLMMMRSEF/logs/%x_%A.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=big
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morteza.ansarinia@uni.lu

########################################################################
# Name:         openneuro_s3.sh
# Description:  This script is used to download OpenNeuro dataset using AWS CLI
#               from OpenNeuro S3 bucket into the shared project folder.
# Requirements: spack (awscli), module (compiler/GCC/12.2.0)
########################################################################

# Install micromamba
spack/bin/spack install awscli
spack/bin/spack load awscli

# NOTE: this is being executed in the shared folder (see --chdir above)
# TODO: copy the commands from the job on the home of the aws cluster
