---
title: "RAM team @ Meta AI"
permalink: /
---
<!-- AUTO-GENERATED from projects/projects.yaml by projects/generate_readme.py. Do not edit by hand: edit projects/projects.yaml and re-run the script. -->
# RAM team @ Meta AI

**View by:** 📂 **Topic** &nbsp;\|&nbsp; [📅 Year](BY_YEAR.md)

Here we list projects undertaken in the RAM framework that are shared publicly, either in the
form of papers, public tasks and/or shared model code. Some of the projects are housed in the
[RAM repo](https://github.com/facebookresearch/RAM/tree/main/projects), others are maintained
via external websites.

See also the [RAM 2 Workshop at COLM 2025](https://facebookresearch.github.io/RAM/workshop/COLM_2025).

**Jump to:** [Autoresearch & Co-Improvement](#autoresearch--co-improvement) · [Agents](#agents) · [Reward Models & Evaluation](#reward-models--evaluation) · [(Self-)Alignment](#self-alignment) · [Synthetic Data](#synthetic-data) · [Pre- and Mid-Training](#pre--and-mid-training) · [Reasoning](#reasoning) · [Memory & Architectures](#memory--architectures)

## Autoresearch & Co-Improvement

- **Aligning Autoresearch** [[paper]](https://arxiv.org/abs/2610.38882) [[tweet]](https://x.com/jaseweston/status/2115992445930311714)
  <em>Humans & AI are better together: studies the human role in autoresearch, and how to evaluate & train aligned co-research agents.</em>

- **HackBench** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/hackbench/) [[dataset]](https://huggingface.co/datasets/facebook/hackbench) [[tweet]](https://x.com/jaseweston/status/2114709989826953719)
  <em>Benchmark for detecting & localizing reward hacking in autoresearch agents; frontier models fail.</em>

- **AutoInteract** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autointeract/) [[tweet]](https://x.com/jaseweston/status/2108176858520924594)
  <em>Trains AIs to be better collaborators via RL with simulated humans in challenging multi-turn interactions.</em>

- **AutoBenchmark** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autobench/) [[tweet]](https://x.com/jaseweston/status/2105305463784935791)
  <em>Agents create benchmarks automatically; human-agent collaboration gives big wins, especially in ideation.</em>

- **Unslopping AI** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/unslop/) [[tweet]](https://x.com/jaseweston/status/2104564368792854860)
  <em>Removes AI slop via RL with rubrics learned from the gap between expert human and model writing (RL-XAR).</em>

- **Autodata** [[paper]](https://arxiv.org/abs/2606.25996) [[blog]](https://facebookresearch.github.io/RAM/blogs/autodata/) [[tweet]](https://x.com/jaseweston/status/2070117091521204521)
  <em>An agentic data scientist that creates high quality data, converting inference compute into better training.</em>

- **AI & Human Co-Improvement** [[paper]](https://arxiv.org/abs/2512.05356) [[tweet]](https://x.com/jaseweston/status/1996773506114101473)
  <em>Position paper: rather than self-improving AI alone, target human-AI co-improvement as the path to safer co-superintelligence.</em>

## Agents

- **Self-Improving Memory** [[blog]](https://facebookresearch.github.io/RAM/blogs/self_improving_memory/) [[tweet]](https://x.com/jaseweston/status/2122691693543817397)
  <em>The model discovers its own memory skills via meta-optimization, then distills them into its weights.</em>

- **Aligning Autoresearch** [[paper]](https://arxiv.org/abs/2610.38882) [[tweet]](https://x.com/jaseweston/status/2115992445930311714)
  <em>Humans & AI are better together: studies the human role in autoresearch, and how to evaluate & train aligned co-research agents.</em>

- **Self-Direction** [[paper]](https://arxiv.org/abs/2610.31846) [[project]](https://self-direction.github.io/)
  <em>Agents turn each mistake into a small LoRA skill, so lessons from failures persist and get reused.</em>

- **AutoInteract** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autointeract/) [[tweet]](https://x.com/jaseweston/status/2108176858520924594)
  <em>Trains AIs to be better collaborators via RL with simulated humans in challenging multi-turn interactions.</em>

- **Experience Synthesis** [[paper]](https://arxiv.org/abs/2511.03773) [[tweets]](https://x.com/jaseweston/status/1986613046047846569)
  <em>Scaling training environments for RL by simulating them with reasoning LLMs.</em>

- **Early Experience** [[paper]](https://arxiv.org/abs/2510.08558) [[tweets]](https://x.com/jaseweston/status/1979179944258265358)
  <em>SFT is sparse; RL on long-horizons is hard. EE provides new mid-training signals that help.</em>

- **Self-Challenging LLM Agents** [[paper]](https://arxiv.org/abs/2506.01716)
  <em>LLM creates own challenging agentic tool-use tasks, resulting in better agentic performance.</em>

- **SWEET-RL** [[project]](https://github.com/facebookresearch/sweet_rl)
  <em>Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks.</em>

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158)
  <em>Generalization to New Tools via Self-Verification.</em>

## Reward Models & Evaluation

- **HackBench** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/hackbench/) [[dataset]](https://huggingface.co/datasets/facebook/hackbench) [[tweet]](https://x.com/jaseweston/status/2114709989826953719)
  <em>Benchmark for detecting & localizing reward hacking in autoresearch agents; frontier models fail.</em>

- **AutoBenchmark** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autobench/) [[tweet]](https://x.com/jaseweston/status/2105305463784935791)
  <em>Agents create benchmarks automatically; human-agent collaboration gives big wins, especially in ideation.</em>

- **Unslopping AI** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/unslop/) [[tweet]](https://x.com/jaseweston/status/2104564368792854860)
  <em>Removes AI slop via RL with rubrics learned from the gap between expert human and model writing (RL-XAR).</em>

- **RLLM** [[paper]](https://arxiv.org/abs/2603.18886) [[blog]](https://facebookresearch.github.io/RAM/blogs/rllm/) [[tweet]](https://x.com/jaseweston/status/2036119252214620513)
  <em>Unified Post-Training via On-Policy-Trained LM-as-RM.</em>

- **HERO** [[paper]](https://arxiv.org/abs/2510.07242) [[tweets]](https://x.com/jaseweston/status/1977756142571864539)
  <em>Combines sparse verifiable and dense RMs into a hybrid reward to give better results.</em>

- **DARLING** [[paper]](https://arxiv.org/abs/2509.02534)
  <em>Method to optimize quality+diversity reward to give gains on each over conventional GRPO RL.</em>

- **StepWiser** [[paper]](https://arxiv.org/abs/2508.19229)
  <em>Stepwise Generative Judge trained with RL. SOTA on ProcessBench; gains when used at train/test time.</em>

- **Reasoning for Factuality** [[paper]](https://www.arxiv.org/abs/2508.05618)
  <em>Shows how to learn CoTs that improve factuality via a new reward function.</em>

- **J1** [[paper]](https://arxiv.org/abs/2505.10320)
  <em>Learns CoTs for LLM-as-a-Judge via GRPO, outperforms EvalPlanner & Distilled R1 models at 8B and 70B scale.</em>

- **Eval-Planner** [[paper]](https://arxiv.org/abs/2501.18099)
  <em>Learning powerful plan+execution CoTs for LLM-as-a-Judge critics, SOTA on RewardBench.</em>

- **Self-Taught Evaluators** [[project]](https://github.com/facebookresearch/RAM/tree/main/projects/self_taught_evaluator)
  <em>Improving LLM-as-a-Judge using iteratively generated synthetic data only (no human annotation).</em>

- **Self-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2401.10020)
  <em>Shows LLMs can judge themselves to self-improve without human feedback.</em>

- **Branch-Solve-Merge** [[paper]](https://arxiv.org/abs/2310.15123)
  <em>Reasoning method to improve LLM Evaluation and Generation.</em>

## (Self-)Alignment

#### _(self-)alignment optimization techniques_

- **AutoInteract** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autointeract/) [[tweet]](https://x.com/jaseweston/status/2108176858520924594)
  <em>Trains AIs to be better collaborators via RL with simulated humans in challenging multi-turn interactions.</em>

- **Unslopping AI** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/unslop/) [[tweet]](https://x.com/jaseweston/status/2104564368792854860)
  <em>Removes AI slop via RL with rubrics learned from the gap between expert human and model writing (RL-XAR).</em>

- **CharacterFlywheel** [[paper]](https://arxiv.org/abs/2603.01973)
  <em>Continual learning in production (with humans-in-the-loop).</em>

- **SPICE** [[paper]](https://arxiv.org/abs/2510.24684) [[tweets]](https://x.com/jaseweston/status/1983343787465150814)
  <em>Challenger creates tasks grounded on documents, Reasoner solves them in self-play, both trained by RL.</em>

- **WaltzRL** [[paper]](https://arxiv.org/abs/2510.08240) [[tweets]](https://x.com/jaseweston/status/1978185306999341256)
  <em>Method to improve safety alignment through multi-agent RL.</em>

- **DARLING** [[paper]](https://arxiv.org/abs/2509.02534)
  <em>Method to optimize quality+diversity reward to give gains on each over conventional GRPO RL.</em>

- **RLHI** [[paper]](https://x.com/jaseweston/status/1972851921255051489) [[tweets]](https://x.com/jaseweston/status/1972851921255051489)
  <em>Method to RL train from organic Human Interaction (aka RLHI) which helps.</em>

- **Bridging Online and Offline RL** [[paper]](https://arxiv.org/abs/2506.21495)
  <em>Mix verifiable & non-verifiable tasks, comparing semi-online DPO & GRPO (similar results).</em>

- **Self-Challenging LLM Agents** [[paper]](https://arxiv.org/abs/2506.01716)
  <em>LLM creates own challenging agentic tool-use tasks, resulting in better agentic performance.</em>

- **Solve & Verify** [[paper]](https://arxiv.org/abs/2502.14948)
  <em>A self-play framework for LLMs to learn how to code by writing code & unit tests.</em>

- **Diversity Preference Optimization** [[paper]](https://arxiv.org/abs/2501.18101)
  <em>SOTA LLMs have model collapse. DivPO training improves diversity with similar quality.</em>

- **Self-Consistency Preference Optimization** [[paper]](https://arxiv.org/abs/2411.04109)
  <em>Self-training without human labels that matches supervised training performance.</em>

- **Thinking LLMs** [[paper]](https://arxiv.org/abs/2410.10630)
  <em>Train LLMs to write down their internal thoughts before responding to general instructions (non-verifiable tasks).</em>

- **Meta-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2407.19594)
  <em>LLMs that can judge their own judgments to self-improve both acting & evaluating actions.</em>

- **Length Following** [[project]](https://github.com/facebookresearch/RAM/tree/main/projects/length_instruct)
  <em>Method to make LLMs follow length instructions much better & remove length bias in evaluations.</em>

- **Iterative Reasoning Preference Optimization** [[paper]](https://arxiv.org/abs/2404.19733)
  <em>Shows how to use iterative DPO to train CoTs on verifiable reasoning tasks.</em>

- **Self-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2401.10020)
  <em>Shows LLMs can judge themselves to self-improve without human feedback.</em>

- **Iterative DPO & Cringe Loss** [[paper]](https://arxiv.org/abs/2312.16682)
  <em>Shows iterative learning improves alignment.</em>

#### _(self-)alignment via other methods_

- **Instruction Back-and-Forth Translation** [[paper]](https://arxiv.org/abs/2408.04614)
  <em>Improves Instruction Backtranslation by rewriting the web document.</em>

- **Instruction Backtranslation** [[paper]](https://arxiv.org/abs/2308.06259)
  <em>Self-Alignment method by predicting instructions for web documents.</em>

- **Leveraging Implicit Feedback** [[paper]](https://arxiv.org/abs/2307.14117)
  <em>Method to learn from human feedback in dialogue deployment data to improve LLM.</em>

#### _data curation_

- **RIP** [[paper]](https://arxiv.org/abs/2501.18578)
  <em>A method to *curate* high quality data, or *create* high quality synthetic data. Gives large improvements.</em>

## Synthetic Data

#### _synthetic data & data quality_

- **Autodata** [[paper]](https://arxiv.org/abs/2606.25996) [[blog]](https://facebookresearch.github.io/RAM/blogs/autodata/) [[tweet]](https://x.com/jaseweston/status/2070117091521204521)
  <em>An agentic data scientist that creates high quality data, converting inference compute into better training.</em>

- **CoT-Self-Instruct** [[paper]](https://arxiv.org/abs/2507.23751)
  <em>Create synthetic data using reasoning followed by filtering for high quality, for large gains.</em>

- **Recycling the Web** [[paper]](https://arxiv.org/abs/2506.04689)
  <em>A method to create more high quality pretraining data via rewriting low quality documents.</em>

- **RIP** [[paper]](https://arxiv.org/abs/2501.18578)
  <em>A method to *curate* high quality data, or *create* high quality synthetic data. Gives large improvements.</em>

- **Instruction Back-and-Forth Translation** [[paper]](https://arxiv.org/abs/2408.04614)
  <em>Improves Instruction Backtranslation by rewriting the web document.</em>

- **Instruction Backtranslation** [[paper]](https://arxiv.org/abs/2308.06259)
  <em>Self-Alignment method by predicting instructions for web documents.</em>

#### _synthetic data for complex reasoning & tools_

- **SPICE** [[paper]](https://arxiv.org/abs/2510.24684) [[tweets]](https://x.com/jaseweston/status/1983343787465150814)
  <em>Challenger creates tasks grounded on documents, Reasoner solves them in self-play, both trained by RL.</em>

- **Self-Challenging LLM Agents** [[paper]](https://arxiv.org/abs/2506.01716)
  <em>LLM creates own challenging agentic tool-use tasks, resulting in better agentic performance.</em>

- **NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions** [[paper]](https://arxiv.org/abs/2502.13124)
  <em>Scaling reasoning capabilities with diverse and high-quality questions.</em>

- **Source2Synth** [[paper]](https://arxiv.org/abs/2409.08239)
  <em>Generating synthetic data from real sources to improve LLMs on complex reasoning tasks.</em>

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158)
  <em>Generalization to New Tools via Self-Verification.</em>

## Pre- and Mid-Training

- **Self-Improving Post-Thinking** [[paper]](https://arxiv.org/abs/2601.21343) [[blog]](https://facebookresearch.github.io/RAM/blogs/post_thinking/) [[tweet]](https://x.com/jaseweston/status/2119885922699546932)
  <em>Teaches models to judge the safety, correctness and confidence of their own generations after answering.</em>

- **Thinking Mid-training: RL of Interleaved Reasoning** [[paper]](https://arxiv.org/abs/2601.21343) [[blog]](https://facebookresearch.github.io/RAM/blogs/thinking_midtraining/) [[tweet]](https://x.com/jaseweston/status/2041864833214095484)
  <em>An intermediate SFT+RL mid-training phase to teach models how to think.</em>

- **Self-Improving Pretraining** [[paper]](https://arxiv.org/abs/2601.21343)
  <em>Uses a strong post-trained model to rewrite pretraining data & judge rollouts, building in safety, factuality & quality early.</em>

- **Recycling the Web** [[paper]](https://arxiv.org/abs/2506.04689)
  <em>A method to create more high quality pretraining data via rewriting low quality documents.</em>

## Reasoning

#### _CoT and RL_

- **Principia** [[paper]](https://arxiv.org/abs/2603.18886) [[blog]](https://facebookresearch.github.io/RAM/blogs/principia) [[tweet]](https://x.com/jaseweston/status/2040062089725645039)
  <em>Reasoning over Mathematical Objects.</em>

- **ParaGator** [[paper]](https://arxiv.org/abs/2603.18886) [[blog]](https://facebookresearch.github.io/RAM/blogs/paragator/) [[tweet]](https://x.com/jaseweston/status/2038609971160424748)
  <em>Train generation with pass@k, and aggregation with pass@1 on-policy, end-to-end for better results.</em>

- **RESTRAIN** [[paper]](https://arxiv.org/abs/2510.02172) [[tweets]](https://x.com/jaseweston/status/1974000962219225271)
  <em>Self-training RL method that improves over other label-free / test-time training methods.</em>

- **AggLM** [[paper]](https://arxiv.org/abs/2509.06870) [[tweets]](https://x.com/jaseweston/status/1965233976949543194)
  <em>Uses RL to train an LLM solution aggregator, with strong results.</em>

- **StepWiser** [[paper]](https://arxiv.org/abs/2508.19229)
  <em>Stepwise Generative Judge trained with RL. SOTA on ProcessBench; gains when used at train/test time.</em>

- **OptimalThinkingBench** [[project]](https://github.com/facebookresearch/RAM/tree/main/projects/otb) [[paper]](https://arxiv.org/abs/2508.13141)
  <em>New benchmark measuring overthinking & underthinking of LLMs.</em>

- **Reasoning for Factuality** [[paper]](https://www.arxiv.org/abs/2508.05618)
  <em>Shows how to learn CoTs that improve factuality via a new reward function.</em>

- **ASTRO** [[paper]](https://arxiv.org/abs/2507.00417)
  <em>Teaching LLMs to reason by reflecting and backtracking in-context.</em>

- **NaturalThoughts** [[paper]](https://arxiv.org/abs/2507.01921)
  <em>Creates better CoT distillation emphasizing difficult and diverse reasoning.</em>

- **Bridging Online and Offline RL** [[paper]](https://arxiv.org/abs/2506.21495)
  <em>Mix verifiable & non-verifiable tasks, comparing semi-online DPO & GRPO (similar results).</em>

- **Thinking LLMs** [[paper]](https://arxiv.org/abs/2410.10630)
  <em>Train LLMs to write down their internal thoughts before responding to general instructions (non-verifiable tasks).</em>

- **Iterative Reasoning Preference Optimization** [[paper]](https://arxiv.org/abs/2404.19733)
  <em>Shows how to use iterative DPO to train CoTs on verifiable reasoning tasks.</em>

#### _other algorithms_

- **SWEET-RL** [[project]](https://github.com/facebookresearch/sweet_rl)
  <em>Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks.</em>

- **Coconut (Continuous Chain-of-Thought)** [[project]](https://github.com/facebookresearch/coconut)
  <em>Training LLMs to reason in continuous latent space (rather than using language).</em>

- **Backtracking Improves Generation Safety** [[paper]](https://arxiv.org/abs/2409.14586)
  <em>Trains LLMs to generate a RESET token if the partial-generation is bad.</em>

- **System 2 Distillation** [[paper]](https://arxiv.org/abs/2407.06023)
  <em>Distilling reasoning traces (System 2) back into the Transformer (System 1).</em>

- **Beyond A\*** [[paper]](https://arxiv.org/abs/2402.14083)
  <em>Better Planning with Transformers via Search Dynamics Bootstrapping.</em>

#### _inference_

- **From decoding to meta-generation** [[paper]](https://arxiv.org/abs/2406.16838)
  <em>Survey paper on reasoning methods.</em>

- **Beyond A\*** [[paper]](https://arxiv.org/abs/2402.14083)
  <em>Better Planning with Transformers via Search Dynamics Bootstrapping.</em>

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158)
  <em>Generalization to New Tools via Self-Verification.</em>

- **System 2 Attention** [[paper]](https://arxiv.org/abs/2311.11829)
  <em>Make LLM plan what it attends to as a generative process, decreasing bias & increasing factuality.</em>

- **Ask, Refine, Trust** [[paper]](https://arxiv.org/abs/2311.07961)
  <em>Technique that uses critical questions to determine if an LLM generation needs refinement.</em>

- **Chain-of-Verification Reduces Hallucination** [[paper]](https://arxiv.org/abs/2309.11495)
  <em>Reduces hallucination by LLM self-identifying and verifying generated facts.</em>

## Memory & Architectures

#### _memory_

- **Self-Improving Memory** [[blog]](https://facebookresearch.github.io/RAM/blogs/self_improving_memory/) [[tweet]](https://x.com/jaseweston/status/2122691693543817397)
  <em>The model discovers its own memory skills via meta-optimization, then distills them into its weights.</em>

- **Self-Direction** [[paper]](https://arxiv.org/abs/2610.31846) [[project]](https://self-direction.github.io/)
  <em>Agents turn each mistake into a small LoRA skill, so lessons from failures persist and get reused.</em>

- **Reverse Training** [[paper]](https://arxiv.org/abs/2403.13799)
  <em>Method for pretraining that helps the reversal curse & improves performance.</em>

- **MemWalker** [[paper]](https://arxiv.org/abs/2310.05029)
  <em>Novel memory architecture: builds & navigates a tree (structured long-term memory) via LLM prompting.</em>

- **Self-Notes** [[project]](https://github.com/facebookresearch/RAM/tree/main/projects/self_notes)
  <em>LLMs generate internal thoughts as they read text, enabling reasoning & memorization.</em>

#### _architectures_

- **Stochastic activations** [[paper]](https://arxiv.org/abs/2509.22358) [[tweets]](https://x.com/jaseweston/status/1972649914233389062)
  <em>Select between several non-linear functions in the feed-forward layers of an LLM.</em>

- **Multi-token attention** [[project]](https://github.com/facebookresearch/RAM/tree/main/projects/mta) [[paper]](https://arxiv.org/abs/2504.00927)
  <em>Attention mechanism that can focus on multiple tokens simultaneously.</em>

- **Byte Latent Transformer** [[paper]](https://arxiv.org/abs/2412.09871)
  <em>New Byte-level LLM architecture that matches tokenization-based LLM performance at scale.</em>

- **Adaptive Decoding via Latent Preference Optimization** [[paper]](https://arxiv.org/abs/2411.09661)
  <em>New layer that selects decoding params automatically *per token*.</em>

- **Contextual Position Encoding** [[project]](https://github.com/facebookresearch/RAM/tree/main/projects/cope)
  <em>New attention mechanism that fixes problems in copying & counting for Transformers.</em>

- **Branch-Train-MiX** [[paper]](https://arxiv.org/abs/2403.07816)
  <em>Novel MoE architecture that is very efficient during training.</em>
