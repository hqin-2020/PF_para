#!/bin/bash

#SBATCH --job-name=pf
#SBATCH --output=pf.out
#SBATCH --error=pf.err
#SBATCH --time=6:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_1.py 