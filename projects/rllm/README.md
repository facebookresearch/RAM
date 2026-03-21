
# Unified Post-Training via On-Policy-Trained Language Model as a Reward Model


## Our Contribution

We develop RLLM, a framework for reinforcement learning (RL) that  **unifies** the post-training paradigm, enabling the policy model to excel across **easy-to-verify, hard-to-verify, and non-verifiable tasks**. 

Reinforcement Learning with an LM as Reward Model (RLLM) first trains an LM-as-RM on on-policy synthetic judgments using RL and  uses its generative rewards to optimize the policy itself. 

The LM-as-RM exploits an LLM's:
- (1) reasoning capabilities to produce higher-quality reward signals and
- (2) instruction-following capabilities to allow flexible reward design.

We show that training the RLLM reward model on-policy (via responses sampled from the policy model) yields **improved results**.


![Method](RLLM.png)

*Figure: The Reinforcement Learning with an LM as Reward Model (RLLM) method compared to standard RLHF and RLVR approaches for post-training LLMs.*


## Why This Matters

Post-training for LLMs typically follows one of two paradigms: Reinforcement Learning from Human Feedback (RLHF), which relies on scalar reward models trained from human preference data, or Reinforcement Learning with Verifiable Rewards (RLVR), which depends on rule-based verifiers. Scalar reward models do not generate chain-of-thought reasoning, making them prone to reward hacking and limiting their effectiveness on complex reasoning tasks. Rule-based verifiers, meanwhile, assume access to gold answers that can be both hard-to-obtain and hard-to-verify, limiting their utility to e.g. easily-verifiable math and code problems. 

We show that **RLLM** can serve as a single, unified post-training recipe, enabling the policy model to excel across easy-to-verify, hard-to-verify, and non-verifiable tasks.

We show that on-policy training of the LM-as-RM outperforms both prompted LMs-as-RMs (including a larger GPT-OSS-120B) and off-policy trained ones. Finally, through extensive analyses across a wide range of policy–reward LM pairings -- varying in model size, capability, and training data (easy- vs. hard-to-verify, reference-free vs. reference-based tasks) -- we identify the key ingredients for effective post-training with Language Models as Reward Models.


## How does it work





## Main Experimental Results

We perform a number of experiments across different settings and backbones for both the LM and the LM-as-RM.
Overall, across all these settings, RLLM achieves consistently higher accuracy and win rates than RLVR and RLHF, with particularly large gains when trained on hard-to-verify problems.


![Method](nonverif.png)

*Figure: Performance comparison of post-trained Qwen3-1.7B models on (a) verifiable tasks (average of five math benchmarks) and (b) non-verifiable instruction-following tasks. Models are trained via RLHF (with \texttt{Skywork-Reward-V2-Llama-3.1-8B} as scalar-RM), RLVR (with \texttt{Math-Verify} as rule-based verifier) and, our RLLM (with \texttt{J1-Qwen3-32B} as LM-as-RM). Post-training data for verifiable tasks is either (1) easy-to-verify, (2) hard-to-verify, (3) reference-free, or (4) reference-based.*

Let's dig a little deeper into the results.


### Reference-free setting
We conduct experiments where we compare  post-trained Qwen3-1.7B (Instruct) models using RLLM or RLHF on easy-to-verify and hard-to-verify reasoning benchmarks in the reference-free setting.
All models are trained on hard-to-verify samples. RLHF'ed models are optimized using SOTA scalar RMs. RLLM models are optimized using either prompted LM-as-RM or our trained \methodrm{} LM-as-RM.
We observe improved RLLM results by scaling up the LM-as-RM, with J1-Qwen3-32B-RM improving AIME24 by 12\% on top of a Qwen3-1.7B (Instruct) model.*

![Method](table1.png)
*Figure: Reference-free setting. RLLM provides strong results compared to RLHF.*

### Reference-based setting
We compare post-trained Qwen3-1.7B (Instruct) models using RLLM or RLVR on easy-to-verify and hard-to-verify reasoning benchmarks in the referenced-based setting. All models are trained on hard-to-verify examples. RLVR models are optimized using either rule-based or model-based verifiers. RLLM models are optimized using either prompted or trained LM-as-RM (functioning as reference-based verifiers). All RLLM variants outperform all RLVR variants.
 
![Method](table2.png)
*Figure: Reference-based setting. RLLM provides strong results compared to RLVR.*

### Easy-to-verify vs. hard-to-verify training sets

Comparison of RLLM, RLHF, and RLVR across different training datasets -- easy-to-verify, hard-to-verify, reference-free, and reference-based. RLLM on hard-to-verify data with a strong LM-as-RM outperforms all models trained on easy-to-verify data.

![Method](table3.png)
*Figure: Reference-based setting}. RLLM provides strong results compared to RLVR.*


![Method](table4.png)
![Method](table5.png)
![Method](table6.png)
![Method](table7.png)

![Method](plot1.png)


## Conclusion



## Contributors
Chenxi Whitehouse, Ilia Kulikov, Ping Yu, Jason Weston, Xian Li, Swarnadeep Saha.

## More details
More details can be found in the [full technical report](https://arxiv.org/abs/2603.18886).

## Citation
If you use our training data or benchmark in your own work, please also cite with the following BibTex entry:
```
@article{principia2026,
  title={Reasoning over mathematical objects: on-policy reward modeling and test time aggregation},
  author={Pranjal Aggarwal, Marjan Ghazvininejad, Seungone Kim, Ilia Kulikov, Jack Lanchantin, Xian Li, Tianjian Li, Bo Liu, Graham Neubig, Anaelia Ovalle, Swarnadeep Saha, Sainbayar Sukhbaatar, Sean Welleck, Jason Weston, Chenxi Whitehouse, Adina Williams, Jing Xu, Ping Yu, Weizhe Yuan, Jingyu Zhang, Wenting Zhao},
  journal={arXiv preprint arXiv:2603.18886},
  year={2026}
}
```
