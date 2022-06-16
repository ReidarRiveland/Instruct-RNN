#!/bin/sh
#SBATCH --job-name=debug_training_launch
#SBATCH --time=00:13:00
#SBATCH --partition=debug-gpu
#SBATCH --array=1
#SBATCH --output=~/worker-logs/debug_training_launch.%a.out
#SBATCH --ntasks=1
#SBATCH --gpus=rtx:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instruct-rnn
python training.py --job_index ${SLURM_ARRAY_TASK_ID} 