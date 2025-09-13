# DARLING
This is the official implementation of the paper "Jointly Reinforcing Diversity and Quality of Language Model Generations".

## Getting Started for Training
Creating conda environment and install dependencies:

    conda create -n verlenv python=3.10
Installing PyTorch (here we only tested on cuda 12.8)

    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
Install other dependencies:

    cd verl
    pip install -e ./
    # This code only uses FSDP. If you need to use megatron please remove USE_MEGATRON=0
    USE_MEGATRON=0  bash scripts/install_vllm_sglang_mcore.sh
    # vllm 0.8.3
    pip install vllm==0.8.3
    # flash-attn
    pip3 install flash-attn --no-build-isolation
To use Wandb, you would also need to set your api key:

    export WANDB_API_KEY=<your_api_key>


## Training Scripts
Training scripts for non-verifiable tasks can be found at `verl/wildchat_scripts`
Training scripts for verifiable tasks can be found at `verl/math_scripts`

## Running Darling
First you would need to serve the partition classifier (an HF checkpoint) from a server:

    bash serve_classifier.sh <PATH_TO_CLASSIFIER_HF>

Then you would need to either manually change the hostname in `verl/verl/utils/reward_score/partition_reward_vllm_serve.py`
or set the environment variable:

    export VLLM_SERVER_HOSTNAME=<your hostname>
Then you can launch `math_scripts/darling.batch`for training on Qwen-4B-Base or `wildchat_scripts/darling.batch` for training wildchat on Llama-3.1-8B-Instruct.

## Evaluation
All evaluation for noveltybench can be found in `evals`.
