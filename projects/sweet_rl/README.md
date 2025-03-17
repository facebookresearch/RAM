# SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks

Official implementation for Collaborative Agent Bench and SWEET-RL.

<p align="center">
| <a href="xx"><b>Paper</b></a> | <a href="https://huggingface.co/datasets/facebook/collaborative_agent_bench"><b>Data</b></a> |
</p>

---

[Yifei Zhou](https://yifeizhou02.github.io/), [Song Jiang](https://songjiang0909.github.io/), [Yuandong Tian](https://yuandong-tian.com/), [Jason Weston](https://ai.meta.com/people/1163645124801199/jason-weston/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Sainbayar Sukhbataar*](https://tesatory.github.io/), [Xian Li*](https://ai.meta.com/people/1804676186610787/xian-li/)
<br>
UC Berkeley, FAIR
<br>
*Equal advising, alphabetical order
[paper_teaser](paper_teaser.png)

## Collaborative Agent Bench
### Quick Start
To set up the environment for Collaborative Agent Bench, run:
```bash
pip install -e .
git clone https://github.com/YifeiZhou02/collab_openrlhf
cd collab_openrlhf
pip install -e .
```
This should have set up the environment for Backend Programming, and it uses a custom fork of openrlhf to support multi-turn DPO and length normalization. 
Optionally, if you also wish to run Frontend Design, you need to install GeckoDriver and Firefox in your system(e.g. https://www.mozilla.org/en-US/firefox/all/desktop-release/ and the command below). 
```bash
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.36.0-linux64.tar.gz
tar -xvzf geckodriver-v0.35.0-linux64.tar.gz
sudo mv geckodriver /usr/local/bin/
```
To verify installation, run:
```bash
geckodriver --version
```


Note that it is possible to install Firefox and GeckoDriver without sudo access by including the path to the applications in ```$PATH``` variable in your system.

To download data, run:
```bash
huggingface-cli download facebook/collaborative_agent_bench colbench_code.jsonl colbench_code_offline_15k_llama8b.jsonl
```

### Testing Your Model on CollaborativeAgentBench
#### Backend Programming

For testing on Backend Programming, you need to first set up an VLLM server as the simulation for human collaborator. To do that, simply run:
```bash
python -m vllm.entrypoints.openai.api_server --model /path/to/llama3.1-70b-instruct --max-model-len 16384 --tensor-parallel-size 8 --gpu-memory-utilization=0.85 --max-num-seqs 16 --port 8000 --enforce-eager --trust-remote-code 
```
Feel free to use llama3.1-8b-instruct as simulator for the human collaborator for reduced gpu memory, but the result may be different from provided in the paper..

After setting up the VLLM server for human collaborator, you can now test your model. For coding, run:
```bash
python scripts/simulate_interactions.py --agent_model /path/to/Llama-3.1-8B-Instruct \
    --hostname xxx or localhost \
    --task_type code \
    --num_tasks 1000 \
    --input_path /path/to/backend_tasks/test.jsonl \
    --output_path /path/for/output/temp_test.jsonl \
    --env_model /path/to/llama3.1-70b-instruct
python scripts/evaluate_code.py /path/for/output/temp_test.jsonl
```
The success rate and the percentage of tests passed will be printed in the end. Note that sometimes LLM generated code might contain print messages, so part of the outputs might be flooded with those messages.
<br>
We also offer a script for you to visualize the trajectories, run:
```bash
python visualizers/visualize_dialogue_histories.py /path/for/output/temp_test.jsonl
```
#### Frontend Design
You can run the following script to download data from WebSight:
```python
from sweet_rl.utils.webpage_utils import replace_urls, render_full_html
import json
from tqdm import tqdm
train_tasks_path = "/your/data/path/frontend_tasks/train.jsonl"
test_tasks_path = "/your/data/path/frontend_tasks/test.jsonl"

from datasets import load_dataset

ds = load_dataset("HuggingFaceM4/WebSight", "v0.2")["train"]


filtered_data = []
for i in tqdm(range(20000)):
    filtered_data.append({
        "problem_description": ds[i]["llm_generated_idea"], 
        "ground_truth": replace_urls(ds[i]["text"]),
    })

with open(train_tasks_path, "w") as f:
    for d in filtered_data[:10000]:
        f.write(json.dumps(d) + "\n")

with open(test_tasks_pathh, "w") as f:
    for d in filtered_data[10000:]:
        f.write(json.dumps(d) + "\n")

```

For testing on Frontend Design, you need to first set up an VLLM server as the simulation for human collaborator. To do that, simply run:
```bash
python -m vllm.entrypoints.openai.api_server --model /path/to/Qwen2-VL-72B-Instruct --max-model-len 16384 --tensor-parallel-size 8 --gpu-memory-utilization=0.85 --max-num-seqs 16 --port 8000 --enforce-eager --limit-mm-per-prompt image=2 --trust-remote-code 
```
Feel free to use Qwen2-VL-7B-Instruct as simulator for the human collaborator for reduced gpu memory, but the result may be different from provided in the paper.


After setting up the VLLM server for human collaborator, you can now test your model for Frontend Design, run:
```bash
python scripts/simulate_interactions.py --agent_model /path/to/Llama-3.1-8B-Instruct \
    --task_type html \
    --num_tasks (100 for fast tests, 500 for paper results) \
    --hostname xxx or localhost \
    --output_path /path/for/output/temp_test_html.jsonl\
    --input_path /path/to/webpage_tasks_all.jsonl \
    --env_model /path/to/Qwen2-VL-72B-Instruct \
python scripts/evaluate_html.py /path/for/output/temp_test_html.jsonl 
```

The average cosine similarity will be printed in the end. We also offer a script for you to visualize the trajectories, run:
```bash
python visualizers/visualize_design_dialogue_histories.py /path/for/output/temp_test_html.jsonl
```

## SWEET-RL (**S**tep-**W**is**E** **E**valuation w/ Training-time information)
Now we provide an example script for running SWEET-RL on Backend Programming. This part assumes that you have set up the environment for Backend Programming.
First set up the paths for loading data and saving intermediate results.
```bash
DATA_PATH=/fsx-ram/yifeizhou/collab_llm/outputs/nov24_train20000_shorter_templatefixed_annotated.jsonl

OUTPUT_DIR=/fsx-ram/yifeizhou/collab_llm/outputs
CHECKPOINT_DIR=/fsx-ram/yifeizhou/collab_llm/checkpoints
```
The intermediate data and checkpoints will be saved to:
```bash
GROUND_TRUTH_PREFERENCES_PATH=$OUTPUT_DIR/temp_ground_truth_preferences.jsonl
REWARD_PATH=$CHECKPOINT_DIR/temp_rm
SAMPLED_PATH=$OUTPUT_DIR/temp_sampled.jsonl
RANKED_PATH=$OUTPUT_DIR/temp_ranked.jsonl
RANDOM_PAIRS_PATH=$OUTPUT_DIR/temp_random_pairs.jsonl
SAVE_PATH=$CHECKPOINT_DIR/temp_dpo
EVALUATION_PATH=$OUTPUT_DIR/temp_evaluation.jsonl
```
We will first train a step-level reward model:
```bash
# first train the step-level reward model with additional training-time information
python scripts/evaluate_code.py $DATA_PATH --k 3 --ground_truth_preference_path $GROUND_TRUTH_PREFERENCES_PATH

deepspeed --module openrlhf.cli.train_dpo \
   --save_path $REWARD_PATH \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --pretrain /PATH/TO/8BLLAMA \
   --bf16 \
   --max_epochs 4 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 2e-7 \
   --beta 0.1 \
   --dataset $GROUND_TRUTH_PATH \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb WANDB_KEY \
   --response_template "<|start_header_id|>assistant<|end_header_id|>" \
   --wandb_run_name sweet_code_rm \
   --mean_log_prob
```
After that, we can use this step-level reward model to generate step-level preference pairs:
```bash
# # Those commands will generate preference pairs given the step-level reward model
python scripts/sample_best_of_n.py $DATA_PATH $SAMPLED_PATH --data_fraction 0.1


python scripts/rank_best_of_n.py --model_id $REWARD_PATH \
    --input_path  $SAMPLED_PATH \
    --output_path $RANKED_PATH 


python scripts/generate_random_pairs_from_ranks.py $RANKED_PATH $RANDOM_PAIRS_PATH --no_prompt --num_pairs 4
```
Finally we can train the model and perform evaluations:
```bash
# # Train the model with step-level preference pairs
deepspeed --module openrlhf.cli.train_dpo \
   --save_path $SAVE_PATH \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps  -1 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --pretrain /PATH/TO/Meta-Llama-3.1-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 16384 \
   --zero_stage 3 \
   --learning_rate 2e-7 \
   --beta 0.1 \
   --dataset $RANDOM_PAIRS_PATH \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --nll_loss_coef 0.01 \
   --use_wandb WANDB_KEY \
   --wandb_run_name sweet_code_8b \



# carry out evaluations
python scripts/simulate_interactions.py --agent_model $SAVE_PATH \
    --hostname host-of-human-simulator \
    --input_path /path/to/backend_tasks/test.jsonl \ \
    --task_type code \
    --num_tasks 1000  --output_path $EVALUATION_PATH

python scripts/evaluate_code.py $EVALUATION_PATH
```
You should be able to see result similar to reported in the paper with a success rate around 40\%.

### Data on Frontend Design
We provide the same command where you can generate the offline data for Frontend Design yourself:
```bash
python scripts/simulate_interactions.py --agent_model /path/to/Llama-3.1-8B-Instruct \
    --task_type html \
    --num_tasks 1000 \
    --best_of_n 6 \
    ---train \
    --hostname xxx or localhost \
    --output_path /path/for/output/temp_test_html.jsonl\
    --input_path /your/data/path/frontend_tasks/train.jsonl \
    --env_model /path/to/Qwen2-VL-72B-Instruct \
    --to_continue
```


## Citation
If you find our benchmark or algorithm useful, please consider citing:






