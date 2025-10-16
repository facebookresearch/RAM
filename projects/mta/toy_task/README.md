## Toy task

These scripts tend to reproduce toy task experiments described in our paper.

### Step 1: Generate data
To generate data for toy task with $N=5$, $L=2$, query=ALL:

```bash
python toy_task/generate_data.py --output_dir <your_path_to_data>/mta_toy_task --block_length 5 --query_legth 2 --train_samples 1000000
```

### Step 2: Train model
To train base model run on a compute node:

```bash
torchrun --nproc-per-node 1 -m train config=./toy_task/configs/find_block_base.yaml
```

Then change seed in configuration file and repeat training two more times to avrage results.

To train MTA model run:

```bash
```

Then change seed in configuration file and repeat training two more times to avrage results.

Results in the paper reported as average over three seeds (42, 43, 44). Exact numbers *will differ on different GPUs*. Results reported in the paper used NVIDIA H100 80GB HBM.

|  | Seed=42 | Seed=43 | Seed=44 | Average |
|----------|----------|----------|----------|----------|
| Base, H100   | 77.7  | 75.3   | 1.9   | 51.6   |
| MTA, H100   | 0   | 0   | 0.2   | 0.1   |
| Base, H200   | 83.9   | 83.1  | 37.6  | 68.2  |
| MTA, H200     | 0.1  | 0.0  | 0.1  | 0.1  |
