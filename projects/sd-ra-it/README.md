# Training LLMs on Self-generated Demonstrations

Scripts and configs for replicating the experiments from ["Post-training an LLM for RAG? Train on Self-Generated Demonstrations"](https://arxiv.org/abs/2502.10596).



You may cite our work as
```bibtex
@misc{finlayson2025posttraining,
      title={Post-training an LLM for RAG? Train on Self-Generated Demonstrations},
      author={Matthew Finlayson and Ilia Kulikov and Daniel M. Bikel and Barlas Oguz and Xilun Chen and Aasish Pappu},
      year={2025},
      primaryClass={cs.CL},
}
```

## Generating self-demos.

1. Obtain training data. We use the training data from the [RA-DIT paper](https://arxiv.org/abs/2310.01352), placed in directories `data/70b/train/tasks.jsonl` and `data/70b/train/oasst.jsonl` with subsampling weights of 0.9 and 0.1.

2. Generate prompts. Use `scripts/prompt_optimization.py`, e.g.,
```sh
python scripts/prompt_optimization.py \
  --dataset_filename "data/70b/tasks.jsonl" \
  --model "Meta-Llama-3-70B-Instruct" \
  --outfile "data/prompts/base.json" \
  --logfile "70B_prompt_optimization.log" \
  --eval_example_count 30 \
  --train_example_count 30 \
  --topk 5 \
  --shuffle_window 400 \
  --beam_size 12 \
  --tensor-parallel-size=2 \
  --chat \
  --steps 5 \
  --rag
```
3. Generate self-demos with `scripts/create_self_demo_train_set.sh`
```sh
bash scripts/create_self_demo_train_set.sh tasks Meta-Llama-3-70B-Instruct
bash scripts/create_self_demo_train_set.sh oasst Meta-Llama-3-70B-Instruct
```

## SFT and DPO training with `fairseq2`

To train a DPO model on self-demonstrations using fairseq2:

```sh
srun fairseq2 lm preference_finetune dpo_checkpoints/fairseq2/self_demo \
  --config-file configs/dpo_70b.yml
```

Other configs correspond to SFT and smaller scale (8B) training runs.

Please refer to documentation on the library setup and examples: https://facebookresearch.github.io/fairseq2/stable/

## Evaluation

1. Obtain eval data with retrievals. We use the evals from the [RA-DIT paper](https://arxiv.org/abs/2310.01352), which comes with retrievals and place them in `data/ra-dit/`.
2. Convert eval files to the correct format.
```sh
python scripts/data/io_to_qas_format.py \
  data/ra-dit/eli5/eli5-dev-kilt.jsonl \
  data/ra-dit/eli5/dev.jsonl
```
3. Run the evaluation.
```sh
judge="Meta-Llama-3.1-405B-Instruct-FP8" # Set to judge model path
eval_set=nq # Set to one of `mmlu zsrequestion conllyagotrunc eli5 hotpotqa nq tqa trex fever wow`
strat=dpo_self_demo_70b # Set to training strategy name
hf_checkpoint= # Set to Huggingface model checkpoint path
pred_tpsize=8 # Tensor parallel size
model_size=70b
ndocs=4
samples=1
preds="data/${strat}/eval/preds/${eval_set}.jsonl"
reward_file="data/${strat}/eval/reward/${eval_set}.jsonl"
reward_file_gemma="data/${strat}/eval/reward/${eval_set}_gemma.jsonl"
response_labels="data/${strat}/eval/response_labels/${eval_set}.jsonl"
response_label_reasons="data/${strat}/eval/response_label_reasons/${eval_set}.jsonl"
relevance="data/relevance/${model_size}/${eval_set}.jsonl"
relevance_reasons="data/relevance_reasons/${model_size}/${eval_set}.jsonl"
resultsfile="results/${strat}/eval/metrics/${eval_set}.json"
datafile=data/ra-dit/${eval_set}/dev.jsonl

# Generate outputs
python scripts/generate.py  \
  --model=${hf_checkpoint} \
  --outfile=$preds \
  --samples=$samples \
  --tensor-parallel-size=$pred_tpsize \
  --ndocs=$ndocs \
  --data $datafile

# Get reward model scores
python scripts/reward_model_gemma.py \
  --outfile=$reward_file_gemma \
  --responses=$preds \
  --ndocs=$ndocs \
  --data $datafile

# Identify whether context contains the answer
python scripts/relevance.py \
  --datafile $datafile \
  --reasoning_file $relevance_reasons \
  --outfile $relevance \
  --ndocs=$ndocs \
  --tensor-parallel-size=$tpsize \
  --judge=$judge \
  --logfile logs/relevance_${eval_set}.log \

# Evaluate (correct/incorrect/refuse) model outputs.
python scripts/eval.py \
  --preds $preds \
  --datafile $datafile \
  --outfile $response_labels \
  --reasoning_file $response_label_reasons \
  --tensor-parallel-size=$tpsize \
  --judge=$judge \
  --logfile logs/response_labels_${eval_set}_${strat}.log \
```
