#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=64G  # cpu memory per gpu
#SBATCH --time=2-00:00:00
#SBATCH --partition=high_priority
#SBATCH --job-name=pitfalls

# Setup distributed args
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "Batch script head node IP: ${MASTER_ADDR} / # nodes: ${SLURM_JOB_NUM_NODES}"

n_proc=1
args="--model gpt --n_layer 6 --n_embd 384 --n_head 6 --n_train 200000 --n_test 20000  --batch_size 256"
args="${args} --dataset graph --deg 2 --path 5 --num_nodes 50 --lr 0.0001 --eval_every 1 --use_wandb"
# args="${args} --prediction_head_sizes 1,2,4,8 --prediction_head_weights 1,1,1,1 --multihead_boundary_condition normalize"
# args="${args} --teacherless"
# args="${args} --reverse"

srun bash -c "torchrun --nproc-per-node ${n_proc} --standalone train.py ${args}"
