#!/bin/bash
#
#SBATCH --job-name=PGQAOA_test
#
#SBATCH --account=fc_vaet
#
#SBATCH --partition=savio
#
#SBATCH --output=PGQAOA_test.log
#
#SBATCH --mail-user=yzb@berkeley.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --nodes=1
#
#SBATCH --time=00:05:00
#
#SBATCH --output=test.out

module purge #Kill all active modules
module load python/3.7
source activate qrl

python SLURM_test.py 