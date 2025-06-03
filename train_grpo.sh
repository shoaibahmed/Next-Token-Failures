#!/bin/bash

args="--model gpt --prediction_head_sizes 1,1,1,1 --prediction_head_weights 1,1,1,1"
args="${args} --n_layer 6 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000  --batch_size 256"
args="${args} --dataset graph --deg 2 --path 10 --num_nodes 50 --lr 0.0001 --waypoint_len 9"
args="${args} --grpo-group-size 16 --grpo-kl-beta 0.1 --use_wandb --save_checkpoints"

python train_grpo.py ${args}
