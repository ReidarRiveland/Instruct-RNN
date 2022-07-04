#!/bin/sh
#SBATCH --job-name=training
#SBATCH --time=05:00:00
#SBATCH --partition=shared-gpu
#SBATCH --output=logs/decoder.%a.out
#SBATCH --ntasks=1
#SBATCH --gpus=ampere:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instructRNN
python training.py 6.20models swap --mode decoder --models sbertNet_tuned --job_index ${SLURM_ARRAY_TASK_ID}