#!/bin/bash

n_proc=1

args="--model next_lat_gpt"
args="${args} --n_layer 6 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000  --batch_size 256"
args="${args} --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.0001 --waypoint_len 4"
args="${args} --use_wandb --save_checkpoints --eval_every 1"

bash -c "torchrun --nproc-per-node ${n_proc} --standalone train.py ${args}"
