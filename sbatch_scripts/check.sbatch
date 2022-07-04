#!/bin/sh
#SBATCH --job-name=check
#SBATCH --time=00:20:00
#SBATCH --partition=shared-gpu
#SBATCH --output=logs/check.0.out
#SBATCH --ntasks=1
#SBATCH --gpus=turing:1
#SBATCH --mem=16GB

module load CUDA
echo $CUDA_VISIBLE_DEVICES

source ~/miniconda3/bin/activate
conda activate instructRNN
python model_checks.py 6.20models swap --mode trained