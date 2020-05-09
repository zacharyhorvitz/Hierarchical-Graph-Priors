#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 48:00:00

# Specify a job name:
#SBATCH -J pixel_pickaxe_log

# Specify an output file
#SBATCH -o pixel_pickaxe_log.out
#SBATCH -e pixel_pickaxe_log.out

# Set up the environment by loading modules
module load cuda/10.2 cudnn/7.6.5

# source venv 
source ../new_venv/bin/activate

# Run a script
unbuffer python train.py --env "pixel_pickaxe_log" --model-type cnn --gpu --seed 5 --lr 0.00025 --batchsize 32 --replay-buffer-size 1000000 --warmup-period 100000 --max-steps 10000000 --test-policy-episodes 20 --reward-clip 0 --epsilon-decay 100000 --model-path ./pixel_pickaxe_log_saved_models --num-frames 1 --address '172.20.201.75' --port 9000

deactivate
