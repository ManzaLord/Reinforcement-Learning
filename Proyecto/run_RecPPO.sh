#!/bin/bash -l
#SBATCH --partition=serial
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --job-name="pesca2"
#SBATCH --output=pesca2.out.%j
#SBATCH --mail-user=esteban.porrasmarin@ucr.ac.cr
#SBATCH --mail-type=END,FAIL

mamba activate pesca_env

python3 Cluster_RecPPO.py
