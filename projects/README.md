
# RAM Projects
Here we list projects undertaken in the RAM framework that are shared publicly, either in the form of papers, public tasks and/or shared model code. This directory also contains subfolders for some of the projects which are housed in the RAM repo, others are maintained via external websites.


## Reasoning

#### _via alignment_

- **Thinking LLMs** [[paper]](https://arxiv.org/abs/2410.10630).
  _Train LLMs to write down its internal thoughts for general instructions (non-verifiable tasks)._

- **Iterative Reasoning Preference Optimization** [[paper]](https://arxiv.org/abs/2404.19733)
  _Shows how to use iterative optimization to train CoTs on verifiable tasks._

#### _other algorithms_

- **Coconut (Continuous Chain-of-Thought)*** [[project]](https://github.com/facebookresearch/coconut).
  _Training LLMs to reason in continuous latent space (rather than using language)._

- **Backtracking Improves Generation Safety** [[paper]](https://arxiv.org/abs/2409.14586).
  _Trains LLMs to generate a RESET token if the partial-generation is bad._

- **System 2 Distillation** [[paper]](https://arxiv.org/abs/2407.06023).
  _Distilling reasoning traces (System 2) back into the Transformer (System 1)._

- **Beyond A*** [[paper]](https://arxiv.org/abs/2402.14083).
  _Better Planning with Transformers via Search Dynamics Bootstrapping._

- **SWEET-RL** [[project]](https://github.com/facebookresearch/sweet_rl).
  _Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks._

#### _inference_

- **From decoding to meta-generation** [[paper]](https://arxiv.org/abs/2406.16838).
  _Survey paper on reasoning methods._

- **System 2 Attention** [[paper]](https://arxiv.org/abs/2311.11829).
  _Make LLM plan what it attends to as a generative process, decreasing bias & increasing factuality._

- **Beyond A*** [[paper]](https://arxiv.org/abs/2402.14083).
  _Better Planning with Transformers via Search Dynamics Bootstrapping._

- **Chain-of-Verification Reduces Hallucination** [[paper]](https://arxiv.org/abs/2309.11495).
  _Reduces hallucination by LLM self-identifying and verifying generated facts._

- **Ask, Refine, Trust** [[paper]](https://arxiv.org/abs/2311.07961).
  _Technique that uses critical questions to determine if an LLM generation needs refinement._

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158).
  _Generalization to New Tools via Self-Verification._

## Evaluation

- **Eval-Planner** [[paper]](https://arxiv.org/abs/2501.18099)).
  _Learning powerful plan+execution CoTs for LLM-as-a-Judge critics, SOTA on RewardBench._

- **Self-Taught Evaluators** [[project]](./self_taught_evaluator).
  _Improving LLM-as-a-Judge using iteratively generated synthetic data only (no human annotation)._

- **Branch-Solve-Merge** [[paper]](https://arxiv.org/abs/2310.15123).
  _Reasoning method to improve LLM Evaluation and Generation._

- **Self-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2401.10020)
  _Shows LLMs can judge themselves to self-improve without human feedback._

## Synthetic Data

#### _synthetic data quality_

- **RIP** [[paper]](https://arxiv.org/abs/2501.18578)
  _A method to *curate* high quality data, or *create* high quality synthetic data. Gives large improvements._

#### _synthetic data for complex reasoning & tools_

- **NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions** [[paper]](https://arxiv.org/abs/2502.13124).
  _Scaling reasoning capabilities with diverse and high-quality questions._

- **Source2Synth** [[paper]](https://arxiv.org/abs/2409.08239).
  _Generating synthetic data from real sources to improve LLMs on complex reasoning tasks._

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158).
  _Generalization to New Tools via Self-Verification._


## (Self-)Alignment

#### _(self-)alignment optimization techniques_

- **Diversity Preference Optimization** [[paper]](https://arxiv.org/abs/2501.18101)
  _SOTA LLMs have model collapse. DivPO training improves diversity with similar quality._

- **Self-Consistency Preference Optimization** [[paper]](https://arxiv.org/abs/2411.04109)
  _self-training without human labels that matches supervised training performance._

- **Thinking LLMs** [[paper]](https://arxiv.org/abs/2410.10630).
  _Train LLMs to write down its internal thoughts before responding to general instructions._

- **Meta-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2407.19594)
  _LLMs that can judge their own judgments to self-improve both acting & evaluating actions._

- **Iterative Reasoning Preference Optimization** [[paper]](https://arxiv.org/abs/2404.19733)
  _Shows how to improve reasoning tasks with iterative DPO._

- **Length Following** [[project]](length_instruct)
  _Method to make LLMs follow length instructions much better & removing length bias in evaluations._

- **Self-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2401.10020)
  _Shows LLMs can judge themselves to self-improve without human feedback._

- **Iterative DPO & Cringe Loss** [[paper]](https://arxiv.org/abs/2312.16682)
  _Shows iterative learning improves alignment._

#### _(self-)alignment via other methods_

- **Instruction Back-and-Forth Translation** [[paper]](https://arxiv.org/abs/2408.04614)
  _Improves Instruction Backtranslation by rewriting the web document._

- **Instruction Backtranslation** [[paper]](https://arxiv.org/abs/2308.06259)
  _Self-Alignment method by predicting instructions for web documents._

- **Leveraging Implicit Feedback** [[paper]](https://arxiv.org/abs/2307.14117)
  _Method to learn from human feedback in dialogue deployment data to improve LLM._

#### _data curation_

- **RIP** [[paper]](https://arxiv.org/abs/2501.18578)
  _A method to *curate* high quality data, or *create* high quality synthetic data. Gives large improvements._

## Memory & Architectures

#### _memory_

- **Reverse Training** [[paper]](https://arxiv.org/abs/2403.13799)
  _Method for pretraining that helps the reversal curse & improves performance._

- **MemWalker** [[paper]](https://arxiv.org/abs/2310.05029)
  _Novel memory architecture: builds & navigates a tree (structured long-term memory) via LLM prompting._

- **Self-Notes** [[project]](self_notes)
  _LLMs generate internal thoughts as they read text, enabling reasoning & memorization._

#### _architectures_

- **Multi-token attention** [[paper]](https://arxiv.org/abs/2504.00927)
  _Attention mechanism that can focus on multiple tokens simultaneously_.

- **Byte Latent Transformer** [[paper]](https://arxiv.org/abs/2412.09871)
  _New Byte-level LLM architecture that matches tokenization-based LLM performance at scale._

- **Adaptive Decoding via Latent Preference Optimization** [[paper]](https://arxiv.org/abs/2411.09661)
  _New layer that selects decoding params automatically *per token*_.

- **Contextual Position Encoding** [[project]](cope)
  _New attention mechanism that fixes problems in copying & counting for Transformers_.

- **Branch-Train-MiX** [[paper]](https://arxiv.org/abs/2403.07816)
  _Novel MoE architecture that is very efficient during training._
