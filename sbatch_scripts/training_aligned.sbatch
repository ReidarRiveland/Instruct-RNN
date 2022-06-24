#!/bin/sh
#SBATCH --job-name=training_aligned
#SBATCH --time=01:30:00
#SBATCH --partition=shared-gpu
#SBATCH --array=0-585
#SBATCH --output=logs/aligned.%a.out
#SBATCH --ntasks=1
#SBATCH --gpus=ampere:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instruct-rnn
python training.py 6.20models aligned --models simpleNet simpleNetPlus comNet comNetPlus sbertNet sbertNet_lin clipNet bertNet bowNet sbertNet_tuned sbertNet_lin_tuned clipNet_tuned bertNet_tuned --job_index ${SLURM_ARRAY_TASK_ID}