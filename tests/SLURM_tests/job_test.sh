#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --output=PGQAOA_test.log

#SBATCH --mail-user=yzb@berkeley.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --nodes=1
##SBATCH --tasks-per-node=32
##SBATCH --constraint=haswell

#SBATCH --job-name=PGQAOA_test

module purge #Kill all active modules
module load python/3.7
source activate qrl

srun python SLURM_test.py 