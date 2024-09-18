#!/bin/sh

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Example set of commands that generates data, trains, and evaluates.

function gen_data {
    nvars=$1 # number of variables

    cmd="python scripts/count_data_gen.py --nvars $nvars --out-path count_data/"

    echo $cmd
    $cmd
}

function launch_train {
    pos_emb=$1 # abs, rel, or cape
    data=$2 # dir containing train/val/test jsonl data
    seed=$3 # random seed

    log_name="${pos_emb}_${data}_seed=${seed}"

    MODEL_ARGS="--nlayers 4 --hid-sz 256 --nheads 4 --block-size 512"
    GENERAL_ARGS="--model simpledec \
        --tokenizer simple \
        --emb-tie \
        --nepochs 100 \
        --drop 0.1 \
        --batch-sz 64 \
        --lr 0.00007 \
        --train-on answer \
        --post-norm \
        --log-plot \
        --log-plot-dir wandb_logs/"

    CUSTOM_ARGS="--data count_data/$data \
        --seed $seed \
        --pos-emb $pos_emb \
        --log-name $log_name \
        --checkpoint checkpoints/${log_name}.pt"

    cmd="python train.py ${GENERAL_ARGS} ${MODEL_ARGS} ${CUSTOM_ARGS}"

    echo $cmd
    $cmd
}

function launch_eval {
    pos_emb=$1 # abs, rel, or cape
    data=$2 # dir containing train/val/test jsonl data
    seed=$3 # seed used for train

    log_name="${pos_emb}_${data}_seed=${seed}"

    cmd="python eval.py --data count_data/$data \
        --checkpoint-path checkpoints/${log_name}.pt \
        --eval-on test"

    echo $cmd
    $cmd
}

set -eo pipefail

gen_data 3
launch_train cope count_var3_step512_train10k 1
launch_eval cope count_var3_step512_train10k 1

launch_train abs count_var3_step512_train10k 1
launch_eval abs count_var3_step512_train10k 1

launch_train rel count_var3_step512_train10k 1
launch_eval rel count_var3_step512_train10k 1


gen_data 5
launch_train cope count_var5_step512_train10k 10
launch_eval cope count_var5_step512_train10k 10

launch_train abs count_var5_step512_train10k 10
launch_eval abs count_var5_step512_train10k 10

launch_train rel count_var5_step512_train10k 10
launch_eval rel count_var5_step512_train10k 10
