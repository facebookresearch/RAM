#!/bin/bash

EXAMPLE_INPUTS=example_inputs.jsonl
MODEL_DIR=facebook/Self-taught-evaluator-llama3.1-70B  # HF repo
EXAMPLE_OUTPUTS=example_outputs.jsonl

python run_model.py --inputs_jsonl_path=$EXAMPLE_INPUTS --model_dir=$MODEL_DIR --results_save_path=${EXAMPLE_OUTPUTS}
