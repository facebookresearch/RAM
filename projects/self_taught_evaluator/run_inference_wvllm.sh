#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

EXAMPLE_INPUTS=./data/eval_data/example_inputs.jsonl
MODEL_DIR=/fsx-ram/shared/Self-taught-evaluator-llama3.1-70B/dpo_model  # CHANGEME
EXAMPLE_OUTPUTS=./example_outputs.jsonl

python src/run_model.py --inputs_jsonl_path=$EXAMPLE_INPUTS --model_dir=$MODEL_DIR --results_save_path=${EXAMPLE_OUTPUTS}
