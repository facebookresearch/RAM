# Contextual Position Encoding (CoPE): Learning to Count What's Important

This code reproduces experiments described in [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/pdf/2405.18719) paper.

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

For example runs to reproduce some of the results reported in the paper for the Counting Task, run the following command:

```bash
bash run.sh
```
