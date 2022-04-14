#!/bin/bash
#
#SBATCH --job-name=AUn1eqp5it2000time #Remember to change accordingly in the python part!!!
#
##SBATCH --cpus-per-task=1#Only use one CPU. Save CPU time. No need for QAOA
#
#SBATCH --output=%x.log #Same as the job name
#
#SBATCH --mail-user=yzb@berkeley.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --nodes=1
#
#SBATCH --time=20:00:00
#
#Specific to NERSC
##SBATCH --time-min=01:00:00
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --tasks-per-node=1
##SBATCH --cpus-per-task=2
##SBATCH --mem=1952


source ~/.bashrc
module purge #Kill all active modules
module load python/3.7-anaconda-2019.10
conda deactivate
conda activate qrl

python /global/homes/l/li87o/codes/PGQAOA_OpenQuantumSystem/train.py --exp_name AUn1eqp5it2000time --path /global/cscratch1/sd/li87o/PGQAOA_results/ --p 5 --num_iters 2000 -lr 1e-2 --testcase cs_au --lr_decay -b 2048 -e 3 --env_dim 1 --fid_fix barrier --au_uni ST --cs_coup eq --fid_adj t
