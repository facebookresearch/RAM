# OptimalThinkingBench: Evaluating Over and Underthinking in LLMs

- OptimalThinkingBench is a unified benchmark that evaluates both overthinking and underthinking in LLMs to encourage development of models that balance performance and efficiency.
- The benchmark consists of two sub-benchmarks: OverthinkingBench with simple queries where long thinking doesn't improve performance, and UnderthinkingBench with complex reasoning tasks where thinking is necessary.
- We introduce thinking-adjusted accuracy metrics (AUCOAA) to measure overthinking and combine it with underthinking accuracy through an F1 score to track optimal thinking.
- Evaluation of 33 models shows no model achieves optimal balance: thinking models generate hundreds of tokens on extremely simple queries without performance gains, while non-thinking models fail on complex reasoning.

<p align="center"><img width="110%" src="figures/otb.png" /></p>

## Paper

This work is based on the following paper: [OptimalThinkingBench: Evaluating Over and Underthinking in LLMs]().

## Setup

The following setup is tested only on h100/h200 gpus. We recommend the below steps to reproduce experiments:

1. Create conda environment

```bash
conda create --name otb python=3.10
conda activate otb
```

2. Install dependencies

```
cd otb_creation
pip install -r requirements.txt
```

3. OptimalThinkingBench Creation

a. Download the Llama-4-Maverick model from [here](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8).

b. Create OverthinkingBench as follows:

```
python create_overthink.py --m path_to_llama_maverick_model
python filter_overthink.py

```

OverthinkingBench will be saved inside `data/overthink_bench_by_maverick.json`

c. Create UnderthinkingBench as follows:

```
python create_underthink.py

```

UnderthinkingBench will be saved inside `data/underthink_data.pkl`

4. Model Evaluation

Coming soon!

## Contributors
Pranjal Aggarwal, Seungone Kim, Jack Lanchantin, Sean Welleck, Jason Weston, Ilia Kulikov, Swarnadeep Saha

## Citation
If you use our benchmark in your own work, please cite with the following BibTex entry:
```
@article{aggarwal2025otb,
  title={OptimalThinkingBench: Evaluating Over and Underthinking in LLMs},
  author={Aggarwal, Pranjal and Kim, Seungone and Lanchantin, Jack and Welleck, Sean and Weston, Jason and Kulikov, Ilia and Saha, Swarnadeep},
  journal={arXiv preprint arXiv: },
  year={2025}
}