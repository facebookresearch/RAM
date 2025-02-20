#!/usr/bin/env sh

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e
eval "$(conda shell.bash hook)"

conda activate pytorch
train_set=$1
model_name=$2
if [ $train_set = "oasst" ]
then
  num=20_000
elif [ $train_set = "tasks" ]
then
  num=200_000
fi
if [ $model_name = "Meta-Llama-3-70B-Instruct" ]
then
  tensor_parallel_size=4
else
  tensor_parallel_size=1
fi
output_file="data/70b/train/${train_set}.jsonl"
mkdir -p $(dirname $output_file)
python scripts/get_demos.py \
  --filename ra-dit/multisource/${train_set}.jsonl \
  --output_file $output_file \
  --n $num \
  --prompts-per-strat 3 \
  --tensor_parallel_size $tensor_parallel_size \
  --model_name $model_name \
  --continued \
  --logfile logs/get_demos_${train_set}_70b.log
