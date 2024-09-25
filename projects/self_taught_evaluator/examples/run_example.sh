#!/bin/bash

HELPSTEER2_VALIDATION=helpsteer2_validation.jsonl
MODEL_DIR=/opt/hpcaas/.mounts/fs-0e3f1457c6d924fc0/kulikov/openeft_fs2_ckpts/wildchat_dpo_it1_public_recipe/checkpoints_hf/step_300_hf
RESULTS_FILEPATH="example_outputs.jsonl"

python run_model.py --inputs_jsonl_path=$HELPSTEER2_VALIDATION --model_dir=$MODEL_DIR --results_save_path=${RESULTS_FILEPATH}
