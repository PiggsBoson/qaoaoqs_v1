#!/bin/bash
#
#SBATCH --job-name=QAOA_array
#SBATCH --array=40
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

exp_name=Had_PGszXmon_LM_n2eqp20T
#SBATCH --output=${exp_name}%a_%A.log

hostname
date
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

script_path=/global/home/users/zhibo_w_yang/codes/qaoaoqs_v1/run/train.py
results_path=/global/scratch/users/zhibo_w_yang/PGQAOA_results/

conda activate qrl

python $script_path --exp_name $exp_name${SLURM_ARRAY_TASK_ID} --path $results_path --p 20 --num_iters 2000 -lr 1e-2 --testcase XmonTLS --env_dim 2 --lr_decay -b 2048 -e 5 --au_uni Had --cs_coup uneq --distribution logit-normal --protocol_renormal True --impl quspin --T_tot $SLURM_ARRAY_TASK_ID --scale 1.0
