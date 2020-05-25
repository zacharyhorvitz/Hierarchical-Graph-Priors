#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 48:00:00

# Specify a job name:
#SBATCH -J skyline_npy

# Specify an output file
#SBATCH -o skyline_npy.out
#SBATCH -e skyline_npy.err

# Set up the environment by loading modules
module load cuda/10.2 cudnn/7.6.5

# source venv 
source ../new_venv/bin/activate

WARMUP=1000000 #added 0
PORT=9000

cd pytorch_dqn/

# Run a script
unbuffer python train.py --env "npy" --run-tag "skyline_npy" --model-type cnn --gpu --seed 5 --lr 0.00025 --batchsize 32 --replay-buffer-size 1000000 --warmup-period $WARMUP --max-steps 10000000 --test-policy-episodes 50 --reward-clip 0 --epsilon-decay $WARMUP --model-path ../skyline_npy_saved_models --num-frames 4 --address '172.25.203.2' --port $PORT --mode skyline

deactivate
