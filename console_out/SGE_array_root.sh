#!/bin/bash
#$ -t 1-7
#$ -cwd             # Current working directory
#$ -N QAOA_array      # Name of submitted job
#$ -S /bin/bash     # Shell for execution
#$ -l h_rt=90:00:00 # Time limit of job in hh:mm:ss
#$ -l mem_free=16G   # Amount of memory requested for job

#$ -j y             # Combine output and error streams into single file

#Doing everything in the root without involving scratch
# Redirect all output to log file
exp_name=Had_PGszXmon_LM_n2eqp20T${SGE_TASK_ID}
exec > ${exp_name}_${JOB_ID}.log 2>&1

hostname
date
echo $SGE_TASK_ID
echo $JOB_ID

script_path=$HOME/qaoaoqs_v1/qaoaoqs/train.py
results_path=$HOME/QAOA_results/tests/

conda activate qrl

python $script_path --exp_name $exp_name --path $results_path --p 20 --num_iters 2000 -lr 1e-2 --testcase XmonTLS --env_dim 2 --lr_decay -b 2048 -e 5 --au_uni Had --cs_coup uneq --distribution logit-normal --protocol_renormal True --impl quspin --T_tot $SGE_TASK_ID --scale 1.0
# Create file to show that job is completed
touch a_COMPLETED_ti${SGE_TASK_ID}.ji$JOB_ID.log

