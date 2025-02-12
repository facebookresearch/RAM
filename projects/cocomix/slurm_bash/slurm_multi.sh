#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=JOB_NAME
#SBATCH -D .

#SBATCH --output=OUTPUT_LOG_PATH
#SBATCH --error=ERROR_LOG_PATH

#SBATCH --account=SLURM_ACCOUNT
#SBATCH --qos=SLURM_QOS
## number of nodes
#SBATCH --nodes=NUM_NODE
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH --cpus-per-task=CPU_NUM
#SBATCH --time=MAX_TIME

# Initialize Conda for bash using Miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate cocomix

# Variables
NUM_PROCESSES=8  # GPUs per node
NUM_MACHINES=$SLURM_JOB_NUM_NODES  # Total nodes
MACHINE_RANK=$SLURM_NODEID  # Rank of the current node
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=12345  # Choose an open port

# Set any necessary environment variables
# export NCCL_DEBUG=INFO
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$MAIN_PROCESS_PORT
export WORLD_SIZE=$(($NUM_PROCESSES * $NUM_MACHINES))
export RANK=$(($MACHINE_RANK * $NUM_PROCESSES))

# Print for debugging
echo "Main process IP: $head_node_ip"
echo "Machine rank: $MACHINE_RANK"
echo "World size: $WORLD_SIZE"

# Print Slurm environment variables
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE"

export LAUNCHER="accelerate launch \
    --config_file ./conf/fsdp_bf16.yaml \
    --num_processes $WORLD_SIZE  \
    --num_machines $NUM_MACHINES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port $MAIN_PROCESS_PORT \
    "
export SCRIPT="main.py"
export SCRIPT_ARGS="$@"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
echo "Running the following command:"
echo $CMD
srun --ntasks=$SLURM_NTASKS $CMD