#!/bin/bash

#SBATCH --account=pi-Def
#SBATCH --job-name=pf3
#SBATCH --output=pf3.out
#SBATCH --error=pf3.err
#SBATCH --time=4:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr

# Load Python modules
module load python/anaconda-2020.02

python Run_Aso1_0_3.py 