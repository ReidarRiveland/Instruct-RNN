#!/bin/sh
#SBATCH --job-name=data_construction
#SBATCH --time=00:13:00
#SBATCH --partition=debug-cpu
#SBATCH --array=1
#SBATCH --output=~/worker-logs/model_training.%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

source ~/miniconda3/bin/activate
conda activate instruct-rnn
python data_loaders/dataset.py 
