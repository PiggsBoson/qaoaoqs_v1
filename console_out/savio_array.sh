#!/bin/bash
#
#SBATCH --job-name=QAOA_array
#SBATCH --array=1,2,3,4,5,6,10,20,30,40,50,60
#
#SBATCH --account=fc_vaet
#
#SBATCH --partition=savio#This partition allows sigle CPU useage
##SBATCH --cpus-per-task=1#Only use one CPU. Save CPU time. No need for QAOA
#
#SBATCH --mail-user=yzb@berkeley.edu
#SBATCH --mail-type=begin,end,fail
#SBATCH --nodes=1
#
#SBATCH --time=72:00:00

exp_name=Kochpp_Had_n0T
#SBATCH --output=${exp_name}%a_%A.log

hostname
date
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

script_path=/global/home/users/zhibo_w_yang/codes/qaoaoqs_v1/run/train.py
results_path=/global/scratch/users/zhibo_w_yang/PGQAOA_results/Koch_paper

source ~/.bashrc
module purge #Kill all active modules
conda init bash
conda deactivate
conda deactivate
conda activate qrl

python $script_path --exp_name $exp_name${SLURM_ARRAY_TASK_ID} --path $results_path --p 20 --num_iters 2000 -lr 1e-2 --testcase Koch_paper_1qb_noLind --env_dim 0 --lr_decay -b 2048 -e 5 --au_uni Had --cs_coup eq --distribution logit-normal --protocol_renormal True --impl quspin --T_tot $SLURM_ARRAY_TASK_ID --scale 1.0

