#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 48:00:00

# Specify a job name:
#SBATCH -J skyline_hier_glove

# Specify an output file
#SBATCH -o skyline_hier_glove.out
#SBATCH -e skyline_hier_glove.err

# Set up the environment by loading modules
module load cuda/10.2 cudnn/7.6.5

# source venv 
source ../../../new_venv/bin/activate

WARMUP=100000 #added 0
PORT=9000

cd .. #pytorch_dqn/

# Run a script
unbuffer python train.py --env "npy" --run-tag "skyline_hier_glove" --gpu --seed 5 --max-steps 1000000 --reward-clip 0 --mode skyline_hier --emb_size 8 --no-tensorboard --use_glove



deactivate
