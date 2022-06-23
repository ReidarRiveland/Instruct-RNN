#!/bin/sh
#SBATCH --job-name=training
#SBATCH --time=04:00:00
#SBATCH --partition=shared-gpu
#SBATCH --array=0-90
#SBATCH --output=logs/training.%a.out
#SBATCH --ntasks=1
#SBATCH --gpus=ampere:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instruct-rnn
python training.py 6.20models swap --models gptXLNet gptXLNet_tuned --job_index ${SLURM_ARRAY_TASK_ID}