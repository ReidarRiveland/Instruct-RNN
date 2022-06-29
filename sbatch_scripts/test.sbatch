#!/bin/sh
#SBATCH --job-name=test
#SBATCH --time=00:30:00
#SBATCH --partition=shared-gpu
#SBATCH --array=0-585
#SBATCH --output=logs/test.%a.out
#SBATCH --ntasks=1
#SBATCH --gpus=ampere:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instruct-rnn
python training.py 6.20models swap --mode test --job_index ${SLURM_ARRAY_TASK_ID}