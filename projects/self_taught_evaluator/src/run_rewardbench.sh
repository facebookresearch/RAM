#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

RB_INPUT=rewardbench_inputs.jsonl
RB_OUTPUT=rewardbench_results.jsonl
MODEL_DIR=/opt/hpcaas/.mounts/fs-0e3f1457c6d924fc0/kulikov/openeft_fs2_ckpts/wildchat_dpo_it1_public_recipe/checkpoints_hf/step_300_hf

python run_model.py --inputs_jsonl_path=$RB_INPUT --model_dir=$MODEL_DIR --prompted_input_jsonl_key="text" --results_save_path=${RB_OUTPUT}

python -m fire utils.py compute_rewardbench_scores --generated_judgements_jsonl=${RB_OUTPUT} --input_prompts_jsonl=${RB_INPUT}
