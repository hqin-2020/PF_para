#!/bin/bash

#SBATCH --job-name=pf2
#SBATCH --output=pf2.out
#SBATCH --error=pf2.err
#SBATCH --time=4:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_2.py 