#!/bin/bash
#$ -t 1-3
#$ -cwd             # Current working directory
#$ -N QAOA_array      # Name of submitted job
#$ -S /bin/bash     # Shell for execution
#$ -l h_rt=90:00:00 # Time limit of job in hh:mm:ss
#$ -l mem_free=16G   # Amount of memory requested for job

#$ -j y             # Combine output and error streams into single file

# usage example: qsub -t 1-3 array_job_bla.sh

# Redirect all output to log file
# exec > job.ti$SGE_TASK_ID.ji$JOB_ID.log 2>&1
exp_name=PG_rn_Heis_e3_LM_n4eqp60T${SGE_TASK_ID}
exec > ${exp_name}_${JOB_ID}.log 2>&1

hostname
date
echo $SGE_TASK_ID
echo $JOB_ID

script_path=$HOME/codes/qaoaoqs_v1/qaoaoqs/train.py
main_dir=$HOME/QAOA_results/PG_new
results_path=/scratch/$USER/QAOA_results/PG_new
task_dir=${results_path}/exp/${exp_name}

source ~/.bashrc
module purge #Kill all active modules
conda init bash
conda deactivate
conda deactivate
conda activate /global/scratch/users/zhibo_w_yang/envs/qrl

conda run -n python $script_path --exp_name $exp_name --path $results_path --p 60 --num_iters 2000 -lr 1e-2 --testcase cs_au --env_dim 4 --lr_decay -b 2048 -e 3 --au_uni ST --cs_coup eq --distribution logit-normal --protocol_renormal True --impl quspin --T_tot $SLURM_ARRAY_TASK_ID --scale 1.0


pyth=$HOME/anaconda/bin/python

# Create file to show that job is completed
touch a_COMPLETED_ti${SGE_TASK_ID}.ji$JOB_ID.log

# Copy output from your scratch directory to your home directory
\rsync -ubav $task_dir ${main_dir}/${exp_name}


# Shred all files in results_path
find $task_dir -depth -type f -exec shred -v -n1 {} \;

rm -rfv $task_dir # Remove output directory

