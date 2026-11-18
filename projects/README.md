<!-- AUTO-GENERATED from projects.yaml by generate_readme.py. Do not edit by hand: edit projects.yaml and re-run the script. -->
# RAM Projects

**View by:** 📂 **Topic** &nbsp;|&nbsp; [📅 Year](BY_YEAR.md)

Here we list projects undertaken in the RAM framework that are shared publicly, either in the form of papers, public tasks and/or shared model code. This directory also contains subfolders for some of the projects which are housed in the RAM repo, others are maintained via external websites.

**Jump to:** [Autoresearch & Co-Improvement](#autoresearch--co-improvement) · [Agents](#agents) · [Reward Models & Evaluation](#reward-models--evaluation) · [(Self-)Alignment](#self-alignment) · [Synthetic Data](#synthetic-data) · [Pre- and Mid-Training](#pre--and-mid-training) · [Reasoning](#reasoning) · [Memory & Architectures](#memory--architectures)

## Autoresearch & Co-Improvement

- **Aligning Autoresearch** [[paper]](https://arxiv.org/abs/2610.38882) [[tweet]](https://x.com/jaseweston/status/2115992445930311714)
  _Humans & AI are better together: studies the human role in autoresearch, and how to evaluate & train aligned co-research agents._

- **HackBench** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/hackbench/) [[dataset]](https://huggingface.co/datasets/facebook/hackbench) [[tweet]](https://x.com/jaseweston/status/2114709989826953719)
  _Benchmark for detecting & localizing reward hacking in autoresearch agents; frontier models fail._

- **AutoInteract** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autointeract/) [[tweet]](https://x.com/jaseweston/status/2108176858520924594)
  _Trains AIs to be better collaborators via RL with simulated humans in challenging multi-turn interactions._

- **AutoBenchmark** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autobench/) [[tweet]](https://x.com/jaseweston/status/2105305463784935791)
  _Agents create benchmarks automatically; human-agent collaboration gives big wins, especially in ideation._

- **Unslopping AI** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/unslop/) [[tweet]](https://x.com/jaseweston/status/2104564368792854860)
  _Removes AI slop via RL with rubrics learned from the gap between expert human and model writing (RL-XAR)._

- **Autodata** [[paper]](https://arxiv.org/abs/2606.25996) [[blog]](https://facebookresearch.github.io/RAM/blogs/autodata/) [[tweet]](https://x.com/jaseweston/status/2070117091521204521)
  _An agentic data scientist that creates high quality data, converting inference compute into better training._

- **AI & Human Co-Improvement** [[paper]](https://arxiv.org/abs/2512.05356) [[tweet]](https://x.com/jaseweston/status/1996773506114101473)
  _Position paper: rather than self-improving AI alone, target human-AI co-improvement as the path to safer co-superintelligence._

## Agents

- **Self-Improving Memory** [[blog]](https://facebookresearch.github.io/RAM/blogs/self_improving_memory/) [[tweet]](https://x.com/jaseweston/status/2122691693543817397)
  _The model discovers its own memory skills via meta-optimization, then distills them into its weights._

- **Aligning Autoresearch** [[paper]](https://arxiv.org/abs/2610.38882) [[tweet]](https://x.com/jaseweston/status/2115992445930311714)
  _Humans & AI are better together: studies the human role in autoresearch, and how to evaluate & train aligned co-research agents._

- **Self-Direction** [[paper]](https://arxiv.org/abs/2610.31846) [[project]](https://self-direction.github.io/)
  _Agents turn each mistake into a small LoRA skill, so lessons from failures persist and get reused._

- **AutoInteract** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autointeract/) [[tweet]](https://x.com/jaseweston/status/2108176858520924594)
  _Trains AIs to be better collaborators via RL with simulated humans in challenging multi-turn interactions._

- **Experience Synthesis** [[paper]](https://arxiv.org/abs/2511.03773) [[tweets]](https://x.com/jaseweston/status/1986613046047846569)
  _Scaling training environments for RL by simulating them with reasoning LLMs._

- **Early Experience** [[paper]](https://arxiv.org/abs/2510.08558) [[tweets]](https://x.com/jaseweston/status/1979179944258265358)
  _SFT is sparse; RL on long-horizons is hard. EE provides new mid-training signals that help._

- **Self-Challenging LLM Agents** [[paper]](https://arxiv.org/abs/2506.01716)
  _LLM creates own challenging agentic tool-use tasks, resulting in better agentic performance._

- **SWEET-RL** [[project]](https://github.com/facebookresearch/sweet_rl)
  _Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks._

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158)
  _Generalization to New Tools via Self-Verification._

## Reward Models & Evaluation

- **HackBench** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/hackbench/) [[dataset]](https://huggingface.co/datasets/facebook/hackbench) [[tweet]](https://x.com/jaseweston/status/2114709989826953719)
  _Benchmark for detecting & localizing reward hacking in autoresearch agents; frontier models fail._

- **AutoBenchmark** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autobench/) [[tweet]](https://x.com/jaseweston/status/2105305463784935791)
  _Agents create benchmarks automatically; human-agent collaboration gives big wins, especially in ideation._

- **Unslopping AI** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/unslop/) [[tweet]](https://x.com/jaseweston/status/2104564368792854860)
  _Removes AI slop via RL with rubrics learned from the gap between expert human and model writing (RL-XAR)._

- **RLLM** [[paper]](https://arxiv.org/abs/2603.18886) [[blog]](https://facebookresearch.github.io/RAM/blogs/rllm/) [[tweet]](https://x.com/jaseweston/status/2036119252214620513)
  _Unified Post-Training via On-Policy-Trained LM-as-RM._

- **HERO** [[paper]](https://arxiv.org/abs/2510.07242) [[tweets]](https://x.com/jaseweston/status/1977756142571864539)
  _Combines sparse verifiable and dense RMs into a hybrid reward to give better results._

- **DARLING** [[paper]](https://arxiv.org/abs/2509.02534)
  _Method to optimize quality+diversity reward to give gains on each over conventional GRPO RL._

- **StepWiser** [[paper]](https://arxiv.org/abs/2508.19229)
  _Stepwise Generative Judge trained with RL. SOTA on ProcessBench; gains when used at train/test time._

- **Reasoning for Factuality** [[paper]](https://www.arxiv.org/abs/2508.05618)
  _Shows how to learn CoTs that improve factuality via a new reward function._

- **J1** [[paper]](https://arxiv.org/abs/2505.10320)
  _Learns CoTs for LLM-as-a-Judge via GRPO, outperforms EvalPlanner & Distilled R1 models at 8B and 70B scale._

- **Eval-Planner** [[paper]](https://arxiv.org/abs/2501.18099)
  _Learning powerful plan+execution CoTs for LLM-as-a-Judge critics, SOTA on RewardBench._

- **Self-Taught Evaluators** [[project]](./self_taught_evaluator)
  _Improving LLM-as-a-Judge using iteratively generated synthetic data only (no human annotation)._

- **Self-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2401.10020)
  _Shows LLMs can judge themselves to self-improve without human feedback._

- **Branch-Solve-Merge** [[paper]](https://arxiv.org/abs/2310.15123)
  _Reasoning method to improve LLM Evaluation and Generation._

## (Self-)Alignment

#### _(self-)alignment optimization techniques_

- **AutoInteract** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/autointeract/) [[tweet]](https://x.com/jaseweston/status/2108176858520924594)
  _Trains AIs to be better collaborators via RL with simulated humans in challenging multi-turn interactions._

- **Unslopping AI** [[paper]](https://arxiv.org/abs/2610.38882) [[blog]](https://facebookresearch.github.io/RAM/blogs/unslop/) [[tweet]](https://x.com/jaseweston/status/2104564368792854860)
  _Removes AI slop via RL with rubrics learned from the gap between expert human and model writing (RL-XAR)._

- **CharacterFlywheel** [[paper]](https://arxiv.org/abs/2603.01973)
  _Continual learning in production (with humans-in-the-loop)._

- **SPICE** [[paper]](https://arxiv.org/abs/2510.24684) [[tweets]](https://x.com/jaseweston/status/1983343787465150814)
  _Challenger creates tasks grounded on documents, Reasoner solves them in self-play, both trained by RL._

- **WaltzRL** [[paper]](https://arxiv.org/abs/2510.08240) [[tweets]](https://x.com/jaseweston/status/1978185306999341256)
  _Method to improve safety alignment through multi-agent RL._

- **DARLING** [[paper]](https://arxiv.org/abs/2509.02534)
  _Method to optimize quality+diversity reward to give gains on each over conventional GRPO RL._

- **RLHI** [[paper]](https://x.com/jaseweston/status/1972851921255051489) [[tweets]](https://x.com/jaseweston/status/1972851921255051489)
  _Method to RL train from organic Human Interaction (aka RLHI) which helps._

- **Bridging Online and Offline RL** [[paper]](https://arxiv.org/abs/2506.21495)
  _Mix verifiable & non-verifiable tasks, comparing semi-online DPO & GRPO (similar results)._

- **Self-Challenging LLM Agents** [[paper]](https://arxiv.org/abs/2506.01716)
  _LLM creates own challenging agentic tool-use tasks, resulting in better agentic performance._

- **Solve & Verify** [[paper]](https://arxiv.org/abs/2502.14948)
  _A self-play framework for LLMs to learn how to code by writing code & unit tests._

- **Diversity Preference Optimization** [[paper]](https://arxiv.org/abs/2501.18101)
  _SOTA LLMs have model collapse. DivPO training improves diversity with similar quality._

- **Self-Consistency Preference Optimization** [[paper]](https://arxiv.org/abs/2411.04109)
  _Self-training without human labels that matches supervised training performance._

- **Thinking LLMs** [[paper]](https://arxiv.org/abs/2410.10630)
  _Train LLMs to write down their internal thoughts before responding to general instructions (non-verifiable tasks)._

- **Meta-Rewarding LLMs** [[paper]](https://arxiv.org/abs/2407.19594)
  _LLMs that can judge their own judgments to self-improve both acting & evaluating actions._

- **Length Following** [[project]](length_instruct)
  _Method to make LLMs follow length instructions much better & remove length bias in evaluations._

- **Iterative Reasoning Preference Optimization** [[paper]](https://arxiv.org/abs/2404.19733)
  _Shows how to use iterative DPO to train CoTs on verifiable reasoning tasks._

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

## Synthetic Data

#### _synthetic data & data quality_

- **Autodata** [[paper]](https://arxiv.org/abs/2606.25996) [[blog]](https://facebookresearch.github.io/RAM/blogs/autodata/) [[tweet]](https://x.com/jaseweston/status/2070117091521204521)
  _An agentic data scientist that creates high quality data, converting inference compute into better training._

- **CoT-Self-Instruct** [[paper]](https://arxiv.org/abs/2507.23751)
  _Create synthetic data using reasoning followed by filtering for high quality, for large gains._

- **Recycling the Web** [[paper]](https://arxiv.org/abs/2506.04689)
  _A method to create more high quality pretraining data via rewriting low quality documents._

- **RIP** [[paper]](https://arxiv.org/abs/2501.18578)
  _A method to *curate* high quality data, or *create* high quality synthetic data. Gives large improvements._

- **Instruction Back-and-Forth Translation** [[paper]](https://arxiv.org/abs/2408.04614)
  _Improves Instruction Backtranslation by rewriting the web document._

- **Instruction Backtranslation** [[paper]](https://arxiv.org/abs/2308.06259)
  _Self-Alignment method by predicting instructions for web documents._

#### _synthetic data for complex reasoning & tools_

- **SPICE** [[paper]](https://arxiv.org/abs/2510.24684) [[tweets]](https://x.com/jaseweston/status/1983343787465150814)
  _Challenger creates tasks grounded on documents, Reasoner solves them in self-play, both trained by RL._

- **Self-Challenging LLM Agents** [[paper]](https://arxiv.org/abs/2506.01716)
  _LLM creates own challenging agentic tool-use tasks, resulting in better agentic performance._

- **NaturalReasoning: Reasoning in the Wild with 2.8M Challenging Questions** [[paper]](https://arxiv.org/abs/2502.13124)
  _Scaling reasoning capabilities with diverse and high-quality questions._

- **Source2Synth** [[paper]](https://arxiv.org/abs/2409.08239)
  _Generating synthetic data from real sources to improve LLMs on complex reasoning tasks._

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158)
  _Generalization to New Tools via Self-Verification._

## Pre- and Mid-Training

- **Self-Improving Post-Thinking** [[paper]](https://arxiv.org/abs/2601.21343) [[blog]](https://facebookresearch.github.io/RAM/blogs/post_thinking/) [[tweet]](https://x.com/jaseweston/status/2119885922699546932)
  _Teaches models to judge the safety, correctness and confidence of their own generations after answering._

- **Thinking Mid-training: RL of Interleaved Reasoning** [[paper]](https://arxiv.org/abs/2601.21343) [[blog]](https://facebookresearch.github.io/RAM/blogs/thinking_midtraining/) [[tweet]](https://x.com/jaseweston/status/2041864833214095484)
  _An intermediate SFT+RL mid-training phase to teach models how to think._

- **Self-Improving Pretraining** [[paper]](https://arxiv.org/abs/2601.21343)
  _Uses a strong post-trained model to rewrite pretraining data & judge rollouts, building in safety, factuality & quality early._

- **Recycling the Web** [[paper]](https://arxiv.org/abs/2506.04689)
  _A method to create more high quality pretraining data via rewriting low quality documents._

## Reasoning

- **RAM 2 Workshop at COLM 2025** [[website]](https://facebookresearch.github.io/RAM/workshop/COLM_2025/)
  _Reasoning, Attention & Memory – 10 Years On: a retrospective on the decade since the first RAM workshop (NeurIPS 2015), and new directions for the next._

#### _CoT and RL_

- **Principia** [[paper]](https://arxiv.org/abs/2603.18886) [[blog]](https://facebookresearch.github.io/RAM/blogs/principia) [[tweet]](https://x.com/jaseweston/status/2040062089725645039)
  _Reasoning over Mathematical Objects._

- **ParaGator** [[paper]](https://arxiv.org/abs/2603.18886) [[blog]](https://facebookresearch.github.io/RAM/blogs/paragator/) [[tweet]](https://x.com/jaseweston/status/2038609971160424748)
  _Train generation with pass@k, and aggregation with pass@1 on-policy, end-to-end for better results._

- **RESTRAIN** [[paper]](https://arxiv.org/abs/2510.02172) [[tweets]](https://x.com/jaseweston/status/1974000962219225271)
  _Self-training RL method that improves over other label-free / test-time training methods._

- **AggLM** [[paper]](https://arxiv.org/abs/2509.06870) [[tweets]](https://x.com/jaseweston/status/1965233976949543194)
  _Uses RL to train an LLM solution aggregator, with strong results._

- **StepWiser** [[paper]](https://arxiv.org/abs/2508.19229)
  _Stepwise Generative Judge trained with RL. SOTA on ProcessBench; gains when used at train/test time._

- **OptimalThinkingBench** [[project]](otb) [[paper]](https://arxiv.org/abs/2508.13141)
  _New benchmark measuring overthinking & underthinking of LLMs._

- **Reasoning for Factuality** [[paper]](https://www.arxiv.org/abs/2508.05618)
  _Shows how to learn CoTs that improve factuality via a new reward function._

- **ASTRO** [[paper]](https://arxiv.org/abs/2507.00417)
  _Teaching LLMs to reason by reflecting and backtracking in-context._

- **NaturalThoughts** [[paper]](https://arxiv.org/abs/2507.01921)
  _Creates better CoT distillation emphasizing difficult and diverse reasoning._

- **Bridging Online and Offline RL** [[paper]](https://arxiv.org/abs/2506.21495)
  _Mix verifiable & non-verifiable tasks, comparing semi-online DPO & GRPO (similar results)._

- **Thinking LLMs** [[paper]](https://arxiv.org/abs/2410.10630)
  _Train LLMs to write down their internal thoughts before responding to general instructions (non-verifiable tasks)._

- **Iterative Reasoning Preference Optimization** [[paper]](https://arxiv.org/abs/2404.19733)
  _Shows how to use iterative DPO to train CoTs on verifiable reasoning tasks._

#### _other algorithms_

- **SWEET-RL** [[project]](https://github.com/facebookresearch/sweet_rl)
  _Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks._

- **Coconut (Continuous Chain-of-Thought)** [[project]](https://github.com/facebookresearch/coconut)
  _Training LLMs to reason in continuous latent space (rather than using language)._

- **Backtracking Improves Generation Safety** [[paper]](https://arxiv.org/abs/2409.14586)
  _Trains LLMs to generate a RESET token if the partial-generation is bad._

- **System 2 Distillation** [[paper]](https://arxiv.org/abs/2407.06023)
  _Distilling reasoning traces (System 2) back into the Transformer (System 1)._

- **Beyond A\*** [[paper]](https://arxiv.org/abs/2402.14083)
  _Better Planning with Transformers via Search Dynamics Bootstrapping._

#### _inference_

- **From decoding to meta-generation** [[paper]](https://arxiv.org/abs/2406.16838)
  _Survey paper on reasoning methods._

- **Beyond A\*** [[paper]](https://arxiv.org/abs/2402.14083)
  _Better Planning with Transformers via Search Dynamics Bootstrapping._

- **ToolVerifier** [[paper]](https://arxiv.org/abs/2402.14158)
  _Generalization to New Tools via Self-Verification._

- **System 2 Attention** [[paper]](https://arxiv.org/abs/2311.11829)
  _Make LLM plan what it attends to as a generative process, decreasing bias & increasing factuality._

- **Ask, Refine, Trust** [[paper]](https://arxiv.org/abs/2311.07961)
  _Technique that uses critical questions to determine if an LLM generation needs refinement._

- **Chain-of-Verification Reduces Hallucination** [[paper]](https://arxiv.org/abs/2309.11495)
  _Reduces hallucination by LLM self-identifying and verifying generated facts._

## Memory & Architectures

- **RAM 2 Workshop at COLM 2025** [[website]](https://facebookresearch.github.io/RAM/workshop/COLM_2025/)
  _Reasoning, Attention & Memory – 10 Years On: a retrospective on the decade since the first RAM workshop (NeurIPS 2015), and new directions for the next._

#### _memory_

- **Self-Improving Memory** [[blog]](https://facebookresearch.github.io/RAM/blogs/self_improving_memory/) [[tweet]](https://x.com/jaseweston/status/2122691693543817397)
  _The model discovers its own memory skills via meta-optimization, then distills them into its weights._

- **Self-Direction** [[paper]](https://arxiv.org/abs/2610.31846) [[project]](https://self-direction.github.io/)
  _Agents turn each mistake into a small LoRA skill, so lessons from failures persist and get reused._

- **Reverse Training** [[paper]](https://arxiv.org/abs/2403.13799)
  _Method for pretraining that helps the reversal curse & improves performance._

- **MemWalker** [[paper]](https://arxiv.org/abs/2310.05029)
  _Novel memory architecture: builds & navigates a tree (structured long-term memory) via LLM prompting._

- **Self-Notes** [[project]](self_notes)
  _LLMs generate internal thoughts as they read text, enabling reasoning & memorization._

#### _architectures_

- **Stochastic activations** [[paper]](https://arxiv.org/abs/2509.22358) [[tweets]](https://x.com/jaseweston/status/1972649914233389062)
  _Select between several non-linear functions in the feed-forward layers of an LLM._

- **Multi-token attention** [[project]](mta) [[paper]](https://arxiv.org/abs/2504.00927)
  _Attention mechanism that can focus on multiple tokens simultaneously._

- **Byte Latent Transformer** [[paper]](https://arxiv.org/abs/2412.09871)
  _New Byte-level LLM architecture that matches tokenization-based LLM performance at scale._

- **Adaptive Decoding via Latent Preference Optimization** [[paper]](https://arxiv.org/abs/2411.09661)
  _New layer that selects decoding params automatically *per token*._

- **Contextual Position Encoding** [[project]](cope)
  _New attention mechanism that fixes problems in copying & counting for Transformers._

- **Branch-Train-MiX** [[paper]](https://arxiv.org/abs/2403.07816)
  _Novel MoE architecture that is very efficient during training._
