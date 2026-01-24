#!/bin/bash

n_proc=1

# Settings taken from: https://arxiv.org/abs/2511.05963 (table 3)
args="--model next_lat_gpt"
args="${args} --n_layer 12 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000"
# args="${args} --dataset graph --deg 7 --path_len 7 --num_nodes 100"
args="${args} --dataset graph --deg 2 --path_len 10 --num_nodes 100"
args="${args} --batch_size 512 --lr 0.0005 --weight_decay 0.1 --epochs 100"
args="${args} --use_wandb --save_checkpoints --eval_every 1"
args="${args} --num_prev_latents 4 --next_latent_pred_layers -1 --use_last_lat_res_conn"

bash -c "torchrun --nproc-per-node ${n_proc} --standalone train.py ${args}"
