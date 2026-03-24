<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Learning to Aggregate through Online Reinforcement Learning


## Our Contribution

We develop RLLM, a framework for reinforcement learning (RL) that **unifies** the post-training paradigm, enabling the policy model to excel across **easy-to-verify, hard-to-verify, and non-verifiable tasks**. 

Reinforcement Learning with an LM as Reward Model (RLLM) first trains an LM-as-RM on on-policy synthetic judgments using RL and  uses its generative rewards to optimize the policy itself. 

The LM-as-RM exploits an LLM's:
- (1) reasoning capabilities to produce higher-quality reward signals and
- (2) instruction-following capabilities to allow flexible reward design.

We show that training the RLLM reward model on-policy (via responses sampled from the policy model) yields **improved results**.


![Method](fig1.png)

*Figure: Our parallel thinking scaffolding and method. We use pass@k optimization for optimizing the initial round of responses and pass@1 optimization (standard RLVR) for optimizing the aggregation rollouts, and train end-to-end.*

![Method](fig2.png)

*Figure: During each round, we sample rollouts from the past aggregation round, pack them into the aggregation prompt, and perform inference to obtain the next pool of rollouts.*




## Why This Matters

## How does it work?

![Method](prompt.png)

*Figure: Aggregation prompt the LLM can use to aggregate its own generations*


## Experimental Results


### Self-aggregation improves frontier models

Parallel generation + aggregation (orange) brings gains across 4 competition math benchmarks (AIME, Brumo, HMMT and IMO-Answerbench) on top of 3 strong models: Kimi-K2-Thinking, Qwen3-4B-Thinking-2507, and
Qwen3-4B-Instruct-2507, compared to standard generation (blue) and majority voting (green).


![Method](lorge.png)
*Figure: Parallel generation + aggregation (orange) brings gains across 4 competition math benchmarks on top of 3 strong models: Kimi-K2-Thinking, Qwen3-4B-Thinking-2507, and
Qwen3-4B-Instruct-2507, compared to standard generation (blue) and majority voting (green).*


### The role of candidate diversity (pass@k) in self-aggregation


![Method](passk.png)
*Figure: Performance of repeated aggregation is upper bounded by the initial pass@k (green) for both Qwen3-4B-Thinking-2507 (left) and Qwen3-4B-Instruct-2507 (right). The asymptotic performance is upper-bounded by the pass@k at the initial round.*

![Method](temp.png)
*Figure: Model = Qwen3-4B-Thinking-2507. Effect of initial sampling temperature on decoding performance, averaged over HMMT, Brumo, and AIME. Increasing the initial temperature leaves pass@1 nearly unchanged while improving pass@k, resulting in higher aggregation performance.*


### Main Experiments


![Method](compare_methods.png)
*Figure: Comparison of training strategies across the initial and aggregation rounds. Columns show whether model parameters are updated via pass@1 or pass@k optimization, or kept fixed.*


#### Competition Math

![Method](main.png)
*Figure: *

![Method](rewards.png)
*Figure: *

#### Scientific Reasoning

![Method](main2.png)
*Figure: *


![Method](rewards2.png)
*Figure: *

## Conclusion

Scaling test-time compute is only as effective as the diversity and quality of the reasoning paths that are explored. Traditional parallel decoding and self-aggregation methods are bottlenecked by off-policy generations and mode collapse. To overcome these limitations, we introduced \method{}, a unified online reinforcement learning framework that explicitly aligns and optimizes candidate generations with downstream aggregation.

Our core insight is that generation and aggregation require distinct but complementary optimization strategies. In \method{}, the generator actively explores a diverse, complementary set of solutions through pass@k optimization. Simultaneously, the aggregator is trained via pass@1 optimization to reliably synthesize the on-policy candidates into a final answer.

Extensive evaluations across competition math and scientific reasoning benchmarks validate the strength of this approach. In both base models (e.g., Qwen3-4B-Base) and strong post-trained reasoners (e.g. Qwen3-4B-Instruct-2507), \method{} consistently improves standard offline self-aggregation. The gains are particularly pronounced on highly complex tasks, such as AIME and Principia, where synthesizing diverse reasoning trajectories is critical. By co-training generation and aggregation end-to-end, \method{} provides a robust, scalable recipe for improving inference-time reasoning.


## Contributors
Tianjian Li, Jingyu Zhang, Ping Yu, Swarnadeep Saha, Sainbayar Sukhbaatar, Jason Weston, Ilia Kulikov, Jack Lanchantin.

## More details
More details can be found in the [full technical report](https://arxiv.org/abs/2603.18886) (see section 3).

## Citation
To reference the work in this blog post, please use the following BibTex entry:
```
@article{principia2026,
  title={Reasoning over mathematical objects: on-policy reward modeling and test time aggregation},
  author={Pranjal Aggarwal, Marjan Ghazvininejad, Seungone Kim, Ilia Kulikov, Jack Lanchantin, Xian Li, Tianjian Li, Bo Liu, Graham Neubig, Anaelia Ovalle, Swarnadeep Saha, Sainbayar Sukhbaatar, Sean Welleck, Jason Weston, Chenxi Whitehouse, Adina Williams, Jing Xu, Ping Yu, Weizhe Yuan, Jingyu Zhang, Wenting Zhao},
  journal={arXiv preprint arXiv:2603.18886},
  year={2026}
}
```
