## Contextual Position Encoding (CoPE): Learning to Count What's Important

- Contextual Position Encoding (CoPE) is a new position encoding method that allows positions to be conditioned on context by incrementing position only on certain tokens determined by the model.
- CoPE allows more general position addressing such as attending to the i-th particular word, noun, or sentence.
- In particular, CoPE computes gate values conditioned on the context first, then uses that to assign positions to tokens using a cumulative sum. This allows positions to be contextualized, and represent the count of different units like words, verbs or sentences. CoPE operates on each attention head and so can attend to different position types on each.

<p align="center"><img width="110%" src="figures/CoPE.png" /></p>

## Papers

This work is based on the following paper: [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/pdf/2405.18719).

## Setup

The following setup is recommened to reproduce experiments:

1. Create conda environment

```bash
conda create --name cope python=3.9
conda activate cope
```

2. Install dependencies:
```bash
conda install pytorch=2.2 pytorch-cuda=12.1 -y --strict-channel-priority --override-channels -c pytorch -c nvidia -c conda-forge
pip install -r requirements.txt
```

## Run model training end evaluation

We created a script that reproduces the results reported in the [paper](https://arxiv.org/pdf/2405.18719) for the Counting Task. Simply run it on a GPU node:

```bash
bash run.sh
```

We reported the average of 3 random seeds.

<p align="center"><img width="110%" src="figures/counting_task.png" /></p>

## Contributors
Olga Golovneva, Tianlu Wang, Janice Lan, Jason Weston, Sainbayar Sukhbaatar
