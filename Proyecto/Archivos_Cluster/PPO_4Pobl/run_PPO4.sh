#!/bin/bash -l
#SBATCH --partition=serial
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --job-name="pesca3"
#SBATCH --output=pesca.out.%j
#SBATCH --mail-user=esteban.porrasmarin@ucr.ac.cr
#SBATCH --mail-type=END,FAIL

mamba activate pesca_env

python3 Cluster_PPO4.py
