#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=16G
#SBATCH -c 6
#SBATCH -t 24:00:00

# Specify a job name:
#SBATCH -J skyline_multitask_2

# Specify an output file
#SBATCH -o sky2_train.out
#SBATCH -e sky2_train.err


# Set up the environment by loading modules
module load python/3.7.4
module load cuda/10.2 cudnn/7.6.5

# source venv 
source ../../new_venv/bin/activate

# Run a script
unbuffer python train.py --env "SKYLINE_2_TASK_TRAIN" --model-type cnn --gpu --seed 5 --lr 0.00025 --batchsize 32 --replay-buffer-size 1000000 --warmup-period 50000 --max-steps 10000000 --test-policy-episodes 20 --reward-clip 0 --epsilon-decay 50000 --model-path ./saved_models_train_2 --num-frames 4 --address '172.25.201.74 ' --port 9001

deactivate
