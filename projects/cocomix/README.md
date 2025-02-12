# CoCoMix

Official PyTorch implementation of "LLM Pretraining with Continuous Concepts".

<p align="center">
    <img src=./cocomix.png width="900"> 
</p>

## Environment
```
conda create -n cocomix python=3.10 -y
conda activate cocomix

# we have developed/tested CoCoMix on torch 2.3.0+cuda12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Code structure

```
Home
|--conf
    |--setup
        |--gpt2_69m_ntp.yaml # config for gpt2 69m pretraining 20B tokens for next token prediction
        |--gpt2_69m_cocomix.yaml # config for gpt2 50m pretraining 20B tokens for cocomix
        |--...
    |--config.yaml # general config for training
    |--config_eval.yaml # general config for evaluation
    |--ddp.yaml # config for huggingface accelerate ddp 
    |--fsdp_bf16.yaml # config for huggingface accelerate fsdp with bf16
|--data
    |--data.py # dataset definition / loader
|--model
    |--sparse_autoencoder
        ... # code for top-k sparse autoencoder
    |--__init__.py # Define model loading, concept extractor loading
    |--concept_extractor.py # GPT2-124M model with SAE
    |--modeling_gpt2_cocomix.py # CoCoMix for GPT2
|--train
    |--train_func
        |--ntp.py # next token prediction
        |--cocomix.py # CoCoMix
    |--trainer.py # trainer function defined: optimizer, scheduler, evaluation
|--main.py # main file, define model, define dataset, define trainer
|--test.py # evaluation functions, we use EleutherAI lm-evaluation-harness
|--utils.py # utility functions: loggers
```

## Preparation and configurations

**dataset**:
- OpenWebText: run `./data/openwebtext_preprocess/prepare.py`. Readme file `./data/openwebtext_preprocess/readme.md`
- Set `data_dir` in `./conf/config.yaml` (e.g., `./data/openwebtext_preprocess`)

**WANDB**: To use weight and bias (wandb) logging
- Create a wandb account and get your wandb key
- Set `wandb_key` in `./conf/config.yaml` as your wandb key
- `wandb_project` in `./conf/config.yaml` is the name of your wandb project
- `wandb_entity` in `./conf/config.yaml` is your wandb entity name
- Set `wandb_log` as false if you don't want to use wandb logging

**Concept related**:
- `insert_layer_index`: Which layer to predict concept labels, insert continous concepts
- `sae_layer_index`: Which layer to extract concepts (from the pretrained model)
- `lam_concept`: concept prediction loss hyperparameter (default: 0.1)
- `concept_dim`: number of concepts on the sparse autoencoder (SAE) latent: pretrained SAE uses 32768 (fixed)
- `concept_num`: number of active concepts (i.e., TopK value of sparse activatation) in TopK SAE: pretrained SAE uses 32 (fixed)

All configuration for next token prediction and cocomix are presented in `./conf/setup/`

## Train code
For all experiments, we have used multi-node training. We have provided a slurm job submit example file in `./slurm_bash`.
- Note that the user needs to fill the details in `./slurm_bash/slurm_multi.sh` to use the slurm file (e.g., account, env_name)
- Currently assuming FSDP (to use DDP, change `--config_file` to `./conf/ddp.yaml`)

We also provide a single-node training example code (without slurm).\
If OOM occurs, please increase the gradient accumulation step `grad_acc_steps` and reduce the micro batch size `update_batch_size`.
```
# train gpt2 69m on openwebtext with next token prediction
sbatch ./slurm_bash/slurm_multi.sh setup=gpt2_69m_ntp

# train gpt2 69m on openwebtext with cocomix
sbatch ./slurm_bash/slurm_multi.sh setup=gpt2_69m_cocomix

# train gpt2 69m on single node with FSDP
accelerate launch --config_file ./conf/fsdp_bf16.yaml --num_processes=8 main.py setup=gpt2_69m_ntp

# train gpt2 69m on single node with DDP
accelerate launch --config_file ./conf/ddp.yaml --num_processes=8 main.py setup=gpt2_69m_ntp 
```

## Evaluation code
Set `data_dir` in `./conf/config_eval.yaml` with the preprocessed openwebtext dataset path (e.g., `./data/openwebtext_preprocess`).\
We use [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) for the evaluation (except for openwebtext validation perplexity). To evaluate on different dataset, please modify `eval_tasks` in `./conf/config_eval.yaml`.\
Note that `eval_single_ckpt` defines whether to evaluate a single checkpoint or evaluate the entire saved checkpoints with a given freqencey (e.g., if the user have saved the ckpt every 2000 training steps, by setting true, it will evaluate all ckpts at once).
```
# two options
# eval_single_ckpt=True or False

# if True, pass the path including the step (e.g., ./logs/.../step_xxx/), this will only evaluate single ckpt 
# the eval_results.json will be saved in ./logs/.../step_xxx/
CUDA_VISIBLE_DEVICES=0 python test.py eval_single_ckpt=True load_path=<LOAD_PATH> 

# else, pass the path excluding the step (e.g., ./logs/.../), this will evaluate all ckpts with a frequency of eval_freq (e.g., step_2000, step_4000, ...)
# the eval_results.json will be saved in ./logs/.../
CUDA_VISIBLE_DEVICES=0 python test.py load_path=<LOAD_PATH> eval_freq=2000
```
