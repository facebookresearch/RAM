
# RAM Projects
Here we list projects undertaken in the RAM framework that are shared publicly, either in the form of papers, public tasks and/or shared model code. This directory also contains subfolders for some of the projects which are housed in the RAM repo, others are maintained via external websites.


## Reasoning

- **Backtracking Improves Generation Safety** [[paper]](https://arxiv.org/abs/2409.14586).
  _Trains LLMs to generate a RESET token if the partial-generation is bad._

- **Self-Taught Evaluators** [[paper]](https://arxiv.org/abs/2408.02666).
  _Improving LLM-as-a-Judge using iteratively generated synthetic data only (no human annotation)._

- **Source2Synth** [[paper]](https://arxiv.org/abs/2409.08239).
  _Generating synthetic data from real sources to improve LLMs on complex reasoning tasks._

- **From decoding to meta-generation** [[paper]](https://arxiv.org/abs/2406.16838).
  _Survey paper on reasoning methods._

- **System 2 Distillation** [[paper]](https://arxiv.org/abs/2407.06023).
  _Distilling reasoning traces (System 2) back into the Transformer (System 1)._

- **System 2 Attention** [[paper]](https://arxiv.org/abs/2311.11829).
  _Make LLM plan what it attends to as a generative process, decreasing bias & increasing factuality._

- **Beyond A*** [[paper]](https://arxiv.org/abs/2402.14083).
  _Better Planning with Transformers via Search Dynamics Bootstrapping._

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158).
  _Generalization to New Tools via Self-Verification._
  
- **Chain-of-Verification Reduces Hallucination** [[paper]](https://arxiv.org/abs/2309.11495).
  _Reduces hallucination by LLM self-identifying and verifying generated facts._
  
- **Branch-Solve-Merge** [[paper]](https://arxiv.org/abs/2310.15123).
  _Reasoning method to improve LLM Evaluation and Generation._

- **Ask, Refine, Trust** [[paper]](https://arxiv.org/abs/2311.07961).
  _Technique that uses critical questions to determine if an LLM generation needs refinement._


  
## Alignment 

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

- **Instruction Back-and-Forth Translation** [[paper]](https://arxiv.org/abs/2408.04614)
  _Improves Instruction Backtranslation by rewriting the web document._ 

- **Instruction Backtranslation** [[paper]](https://arxiv.org/abs/2308.06259)
  _Self-Alignment method by predicting instructions for web documents._ 

- **Leveraging Implicit Feedback** [[paper]](https://arxiv.org/abs/2307.14117)
  _Method to learn from human feedback in dialogue deployment data to improve LLM._


## Memory & Architectures

- **Contextual Position Encoding** [[project]](cope)
  _New attention mechanism that fixes problems in copying & counting for Transformers_

- **Branch-Train-MiX** [[paper]](https://arxiv.org/abs/2403.07816)
  _Novel MoE architecture that is very efficient during training._
  
- **Reverse Training** [[paper]](https://arxiv.org/abs/2403.13799)
  _Method for pretraining that helps the reversal curse & improves performance._

- **MemWalker** [[paper]](https://arxiv.org/abs/2310.05029)
  _Novel memory architecture: builds & navigates a tree (structured long-term memory) via LLM prompting._

- **Self-Notes** [[paper]](https://arxiv.org/abs/2305.00833)
  _LLMs generate internal thoughts as they read text, enabling reasoning & memorization._
