#!/bin/bash
#
#SBATCH --job-name=AU20a #Remember to change accordingly in the python part!!!
#
#SBATCH --account=fc_vaet
#
#SBATCH --partition=savio#This partition allows sigle CPU useage
##SBATCH --cpus-per-task=1#Only use one CPU. Save CPU time. No need for QAOA
#
#SBATCH --output=%x.log #Same as the job name
#
#SBATCH --mail-user=yzb@berkeley.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --nodes=1
#
#SBATCH --time=50:00:00

source ~/.bashrc
module purge #Kill all active modules
module load python/3.7
conda init bash
conda deactivate
conda activate /global/scratch/zhibo_w_yang/envs/qaoa

python /global/home/users/zhibo_w_yang/codes/PGQAOA_OpenQuantumSystem/train.py --exp_name AU20a --path /global/scratch/zhibo_w_yang/PGQAOA_results/ --p 30 --num_iters 3000 -lr 1e-2 --testcase cs_au --lr_decay -b 2048 -e 3 --env_dim 6 --fid_fix barrier --au_uni ST