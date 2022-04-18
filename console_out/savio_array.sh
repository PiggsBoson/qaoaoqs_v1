#!/bin/bash
#
#SBATCH --job-name=QAOA_array
#SBATCH --array=40
#SBATCH --output=PG_rn_Heis_e3_LM_n4eqp60T%a_%A.log
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

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

source ~/.bashrc
module purge #Kill all active modules
conda init bash
conda deactivate
conda deactivate
conda activate /global/scratch/users/zhibo_w_yang/envs/qrl

python /global/home/users/zhibo_w_yang/codes/PGQAOA_OpenQuantumSystem/train.py --exp_name PG_rn_Heis_e3_LM_n4eqp60T$SLURM_ARRAY_TASK_ID --path /global/scratch/users/zhibo_w_yang/PGQAOA_results/PG_new --p 60 --num_iters 2000 -lr 1e-2 --testcase cs_au --env_dim 4 --lr_decay -b 2048 -e 3 --au_uni ST --cs_coup eq --distribution logit-normal --protocol_renormal True --impl quspin --T_tot $SLURM_ARRAY_TASK_ID --scale 1.0

