#!/bin/sh
#SBATCH --job-name=training_big
#SBATCH --time=25:00:00
#SBATCH --partition=public-gpu
#SBATCH --array=0-90
#SBATCH --output=logs/aligned_big.%a.out
#SBATCH --ntasks=1
#SBATCH --gpus=turing:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instruct-rnn
python training.py 6.20models aligned --models gptNetXL gptNetXL_tuned --job_index ${SLURM_ARRAY_TASK_ID}