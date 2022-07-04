#!/bin/sh
#SBATCH --job-name=training
#SBATCH --time=02:00:00
#SBATCH --partition=shared-gpu
#SBATCH --output=logs/training.%a.out
#SBATCH --ntasks=1
#SBATCH --gpus=ampere:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instructRNN
python training.py 6.21models swap --job_index ${SLURM_ARRAY_TASK_ID}