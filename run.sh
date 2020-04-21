#!/usr/bin/env bash
python train.py \
--env "PongNoFrameskip-v4" \
--model-type cnn \
--gpu \
--seed 5 \
--lr 0.00025 \
--batchsize 32 \
--replay-buffer-size 100000 \
--warmup-period 10000  \
--max-steps 200000 \
--test-policy-episodes 5 \
--reward-clip 1 \
--epsilon-decay 10000 \
--model-path ./saved_models
