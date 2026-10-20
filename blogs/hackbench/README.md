---
title: "HackBench: Detecting and Localizing Reward Hacking in Autoresearch Agents"
permalink: /blogs/hackbench/
---

<style>
.hackbench-figure {
  overflow-x: auto;
  margin: 1rem auto;
  text-align: center;
  -webkit-overflow-scrolling: touch;
}

.hackbench-figure img {
  display: block;
  width: var(--figure-width, 100%);
  max-width: 100%;
  height: auto;
  margin-inline: auto;
  cursor: zoom-in;
}

.hackbench-figure-hint {
  display: none;
}

@media (max-width: 700px) {
  .hackbench-figure img {
    width: var(--mobile-width, 760px);
    max-width: none;
    margin-inline: 0;
  }

  .hackbench-figure-hint {
    display: block;
    margin-top: -0.5rem;
    color: #57606a;
    font-size: 0.85rem;
  }
}
</style>

# HackBench: Detecting and Localizing Reward Hacking in Autoresearch Agents

<a id="fig-hackbench-located-hack"></a>

<div class="hackbench-figure" style="--figure-width: 89%;">
  <a href="fig0.png"><img src="fig0.png" alt="Figure 1: HackBench paper figure" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Figure 1. **HackBench.** Located hacks per model. All the systems measured have many hack detection failures._

## Overview

**Motivation**

Language model agents can carry out long-horizon research tasks end to end: given a task (“pretrain a language model with lowest dev perplexity,” “finetune a classifier to achieve highest possible accuracy”), the agent reads the code base, writes and runs training and evaluation code, and is scored by an automated *verifier* ([Chan et al., 2024][chan2024mlebench]; [Wijk et al., 2024][wijk2024rebench]; [Starace et al., 2025][starace2025paperbench]; [Huang et al., 2024][huang2024mlagentbench]). However, this optimization loop can be fragile. Even when the score is computed on held-out data in a separate environment, the agent knows, or can infer, how the score is computed: it can read the evaluation code, inspect the test labels or test distribution, and shape its solution to the evaluation metric rather than to the task itself. When the objective is “make this metric go up,” an agent that did not conduct the underlying research can still make the number go up by *reward hacking*: exploiting a flaw in the specification to obtain reward without accomplishing the intended task ([Skalse et al., 2022][skalse2022defining]; [Pan et al., 2022][pan2022effects]; [Weng, 2024][weng2024rewardhack]).

The community has highlighted the importance and difficulty of dealing with reward hacking in autoresearch: PostTrainBench re-audited its leaderboard after agents built training data around held-out items, distilled from external model APIs, and submitted instruction-tuned checkpoints in place of the assigned base model ([Team, 2026][posttrainbench2026v11]). On NanoGPT speedrun ([Jordan et al., 2024][modded_nanogpt_2024]), METR observed agents freezing weights or halting training the moment the target loss was touched, and needed the speedrun's human maintainer to separate these brittle gains from legitimate optimization ([Cunningham et al., 2026][metr2026expenditure]). In each case the hack is task-specific, looks like ordinary progress at the level of individual actions, and surfaces only in long transcripts. The cost of missing it is high: autoresearch verifiers are exactly the reward signals used to train agents with reinforcement learning from verifiable rewards ([DeepSeek-AI et al., 2025][deepseekai2025deepseekr1incentivizingreasoningcapability]; [Lightman et al., 2024][lightman2024lets]), so a hack that survives corrupts the training signal ([Qi et al., 2026][anthropic2026rewardseeker]).

A natural research question arises: can an LLM *monitor* catch these hacks by reading the agent's trajectory -- i.e., *can LLMs police themselves*? This task is particularly challenging: A trajectory can run to millions of tokens, so a monitor must first exercise long-context *memory*; the hack can be a handful of commands among hundreds (a median of 264 per trajectory in the benchmark we create in this work), frequently buried mid-trajectory where attention is weakest ([Liu et al., 2023][liu2023lost]; [Hsieh et al., 2024][hsieh2024ruler]). It must then *reason* across steps, because a hack is often a multi-step action whose sub-steps are distributed through the transcript, and exact localization requires reproducing and grounding the exact command rather than merely noticing that something is off. Finally, the monitor must bring task *knowledge*: deciding whether a command games a grader is a scientific judgment about the task, not a surface pattern. Unlike a synthetic needle-in-a-haystack, the needle here cannot be found by string-matching or by spotting an inconsistency, only by understanding it ([Baker et al., 2025][baker2025monitoring]). Fundamentally, even if detectable, this comes down to a question of alignment with humans: *even if this action can improve the reward, is this a behavior humans would consider acceptable?*


**Our contribution**

To measure how well LLMs catch reward hacking, we build HackBench, a benchmark for detecting and localizing reward hacks in full autoresearch trajectories. HackBench is built from *real* agent runs and includes both *natural* and *synthetic* hacks. The *synthetic* hacks are long-context stress tests: covert, often multi-step hacks are injected into long honest trajectories, and causally affected downstream observations are patched so that the needle remains consistent with world state. Because we know exactly which commands were injected, these trajectories come with ground-truth labels for scoring both detection and localization.

Using HackBench, we evaluate the following frontier proprietary models: Claude Opus 5, Claude Opus 4.8, GPT-5.6 Sol, GPT-5.5, Muse Spark 1.1, Muse Spark 1.2 and Muse Spark 1.3 and the following open-sourced models: GLM 5.2, Kimi-K3, Kimi-K2.6, Qwen3.5-397B-A17B, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B, under multiple settings: binary verdict (hack or legitimate), and command localization tasks.

Our contributions are:

- [HackBench](https://huggingface.co/datasets/facebook/hackbench), an open-source benchmark of 202 long-context autoresearch trajectories for reward-hack detection and localization, with human-audited command-level gold labels and three task formats;
- a diagnosis of where LLM monitors fail at detection and localization, organized along the long-context *memory*, multi-step *reasoning*, and task *knowledge* axes the problem demands;
- a study of monitor scaffolds and test-time-compute scaling, showing that long-context management is essential.

<a id="fig-teaser"></a>

<div class="hackbench-figure" style="--figure-width: 80%;">
  <a href="fig1.png"><img src="fig1.png" alt="Figure 2: HackBench paper figure" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Figure 2. HackBench: Given a full agent trajectory (left), an LLM monitor (center) reads the entire transcript and is scored on two tasks (right): binary detection, labeling the run HACK vs. LEGIT, and command localization, returning the exact offending command(s)._

## HackBench

HackBench evaluates how well an LLM monitor can detect and localize reward hacking from a full autoresearch trajectory. It combines *real* audited trajectories with a *synthetic* long-context stress test, allowing verification via human-audited command-level gold labels. This section formalizes the problem, describes where the trajectories come from and our agent settings, details the construction of the real and injected sets, and catalogs what the hacks look like.

### Problem formulation

**Autoresearch tasks and trajectories.** An autoresearch task provides an agent with a working environment (code, data, tools) and an instruction to optimize a metric. The agent interacts with the environment over a long horizon, producing a *trajectory* τ = (s_0, r_1, a_1, o_1, r_2, a_2, o_2, …, r_T, a_T, o_T), where each step contributes the agent's private reasoning r_t, a command a_t (keystrokes issued to a shell), and an observation o_t (terminal output). At the end, a *verifier* V runs in the environment and returns a scalar reward R = V(env_T), typically by recomputing a metric from artifacts the agent produced.

**Reward Hacking.** A trajectory contains *reward hacking* if the agent obtains reward by subverting the measurement rather than by accomplishing the task, e.g. writing fabricated metric files, editing or monkey-patching the grader, training or selecting on held-out test data, or reporting a subset of results chosen to inflate the aggregate. We contrast this with *legitimate* work, including genuine iteration, using pretrained models, or cutting compute while still doing the real task. The distinction is a property of *how* the reward was obtained, so it can only be judged from the trajectory, not from R alone.

**Monitoring.** A *monitor* M maps a trajectory (and optionally the task and verifier) to a judgment about reward hacking. A monitor reads a completed trajectory τ and outputs a verdict and, optionally, a localization of the specific command(s) that constitute the hack.


<div class="hackbench-figure" style="--figure-width: 79%; --mobile-width: 900px;">
  <a href="table1.png"><img src="table1.png" alt="Table 1: taxonomy" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Table 1. A taxonomy of reward hacks observed in autoresearch trajectories that we use for HackBench, grouped by scope (general vs. task-specific), with one example per category. General hacks apply across tasks; task-specific hacks exploit the measurement structure of a particular task. Examples are drawn from ResearchGym and AutoLab trajectories._

**Task framing** We evaluate monitors under three increasingly strict framings:

- **Binary detection**: the monitor emits a HACK/LEGIT verdict. We report F₁ and recall for the HACK class, together with balanced accuracy.

- **Numbered localization**: each executable command is prefixed with a `[CMD n]` index, and the monitor returns the indices of the hacking commands.

- **Verbatim localization**: the monitor returns the hacking commands verbatim from the transcript.

For both localization framings, metrics are computed over HACK trajectories. We report *located-any*, indicating whether at least one gold command is identified; *exact match*, indicating whether the predicted command set exactly equals the gold set; and command-set Jaccard similarity. Numbered localization is *recognition* (emit the indices of hack commands); verbatim localization additionally requires *reproduction and grounding* (quote the exact line, verified as a real substring), which is strictly harder. Localizing a hack in a long trajectory requires retaining relevant context, reasoning across multiple steps, and understanding the task and its grading procedure. HackBench therefore also evaluates long-context reasoning in a semantically meaningful setting.

### Autoresearch Grounding

<!--
<details markdown="1">
<summary>See task suite and agent setup</summary>
-->

**Tasks.** Our trajectories come from long-horizon ML research tasks. Our task suite comprises 5 tasks from ResearchGym ([Garikaparthi et al., 2026][garikaparthi2026researchgymevaluatinglanguagemodel]) and 7 tasks from the AutoLab “Model Development” subset ([Xu et al., 2026][autolab-2026]). Additionally, we include one frontier autoresearch task on a real research codebase: automated-w2s-research ([Wen, 2026][safetyresearch2026automatedw2s]).

**Autoresearch Agents** We drive the agent with the Harbor terminus-2 scaffold ([Team, 2026][Harbor_Framework]), in which the agent has access only to a bash tool: it issues shell keystrokes and observes terminal output, with each task running in a container. We deliberately did not use more advanced scaffolds because the goal of this project is not to improve autoresearch performance, and a simple terminal provides a clean testbed for studying reward hacking. We use GPT-5.5 ([OpenAI, 2026][openai2026gpt55]) and GLM-5.2 ([Z.ai, 2026][zai2026glm52]) to generate research trajectories.

<!--
</details>
-->

### HackBench Construction

In our preliminary experiments, agents rarely hacked when the instruction explicitly prohibited reward hacking, and the hacks we observed only covered a few types (e.g. fabrication, cherry-picking random seeds). Thus, while we collect naturally occurring hacks, collecting enough across a broad range of types was therefore impractical. We thus developed a taxonomy of hacks (Table 1) and used it to guide synthetic hack injection into agent transcripts. Synthetic injection provides ground-truth labels for the injected commands and lets us control each hack's type and location.

**Injecting Synthetic Hacks.** We construct synthetic hack trajectories in three stages (Figure 3). First, we sample an agent trajectory containing bash commands, chain-of-thought reasoning, and terminal observations (Step 1). Second, we sample a hack type from our taxonomy and insert commands that implement it into the otherwise unmodified autoresearch trajectory (Step 2). Third, we update downstream observations to reflect the injected commands' effects, changing only affected state information while preserving the task, original commands, and unrelated observations (Step 3). We used GLM 5.2 to perform both command injection and observation updates. We then used Claude Opus 4.8 and GPT-5.5 to audit each trajectory to check whether the injected commands are reward hacking and whether downstream observations are consistent with the injection. We manually annotated cases where the two auditors disagreed.

**Collecting Natural Hacks.** To surface candidates for naturally occurring hacks, we used GPT-5.6 Sol to screen standard autoresearch trajectories, dividing each transcript into eight chunks. We selected this configuration because it achieved the highest HACK recall in our monitor evaluations, including on the challenging synthetically injected subset.

After auditing the candidates, we constructed HackBench with 202 trajectories: 41 containing *natural* hacks, 128 containing *synthetic* hacks, and 33 legitimate trajectories.

<a id="fig-injection_pipeline"></a>

<div class="hackbench-figure" style="--figure-width: 85%;">
  <a href="fig2.png"><img src="fig2.png" alt="Figure 3: HackBench paper figure" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Figure 3. Construction of a HackBench example with a *synthetic* hack through trajectory editing. Step 1: We begin with a real autoresearch trajectory. Step 2: We insert a synthetic reward-hacking command (red). Step 3: We update only causally affected downstream observations (green), preserving the task, commands, and unrelated observations. This produces a coherent example with command-level human-audited gold annotations._

## Binary Hack Detection Results


<div class="hackbench-figure" style="--figure-width: 79%; --mobile-width: 900px;">
  <a href="table2.png"><img src="table2.png" alt="Table 2: results" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Table 2. Binary reward-hacking detection performance (%) on HackBench (n=202; 169 HACK and 33 LEGIT). Reported values are mean ± sample standard deviation across three complete independent trials. We used the default reasoning effort with temperature 1.0 throughout our experiments._

<details markdown="1">
<summary>See evaluation setup</summary>

### Setup

We evaluate the following proprietary models: Claude Opus 5, Claude Opus 4.8, GPT-5.6 Sol, GPT-5.5, and Muse Spark 1.1, Muse Spark 1.2 and Muse Spark 1.3, as well as the following open-weight models: GLM-5.2, Kimi K3, Kimi K2.6, Qwen3.5-397B-A17B, Qwen3.5-122B-A10B, and Qwen3.5-35B-A3B. Each model acts as a monitor that reads an autoresearch transcript and outputs a binary HACK/LEGIT verdict. Unless explicitly stated, we used the default reasoning effort for all models (i.e. passing no reasoning effort parameter).  We report F₁, recall on the HACK class, as well as balanced accuracy under two settings:

**Single Pass**: the entire transcript is placed in the monitor's context, with the tail truncated when the transcript exceeds the context window.

**Chunked**: the transcript is split into 8 chunks that are judged in parallel by independent monitors, and the 8 chunk-level verdicts are aggregated into a final verdict. The aggregator can choose to accept the chunk-level HACK verdicts, or overturn to LEGIT based on its judgement.

</details>

### Binary Detection Results

**Binary hack detection is hard, even for frontier models.** Table 2 reports overall performance. In the single-pass setting, GPT-5.6 Sol achieves the highest F₁ (80.96) and HACK recall (79.68%), but its balanced accuracy is only 45.90%. Because HACK trajectories account for 83.7% of the benchmark, the Always HACK baseline achieves an F₁ of 91.11, exceeding every evaluated monitor in both settings. High F₁ can be achieved by always leaning towards HACK. Muse Spark 1.2 achieves the highest balanced accuracy in both settings, at 73.99% with single-pass monitoring and 71.99% with chunking. Several single-pass monitors also miss most hacks: Claude Opus 4.8, Kimi K2.6, and GLM-5.2 have recall below 34%, while all three Qwen3.5 models have recall below 7%. The substantial improvements these models obtain from chunking suggest that limited long-context access contributes to their poor single-pass performance.

**Chunking substantially improves detection.** Chunking improves F₁ and HACK recall for all 13 monitors. The largest F₁ gains are achieved by Qwen3.5-35B-A3B (+56.38 points, from 11.85 to 68.23) and GLM-5.2 (+51.22 points, from 26.56 to 77.78). GPT-5.6 Sol remains the strongest chunked monitor by F₁ (85.60) and HACK recall (86.19%), while Kimi K2.6 achieves the highest chunked F₁ among the open-weight models (84.92). All four models with 256K context windows gain between 48.79 and 56.38 F₁ points, consistent with difficulty detecting hacks in long transcripts. However, GLM-5.2 also benefits substantially despite its 1M context window, suggesting that chunking helps beyond merely avoiding context truncation.

<a id="fig-dumbell_breakdown_by_types"></a>

<div class="hackbench-figure" style="--figure-width: 89%;">
  <a href="fig3.png"><img src="fig3.png" alt="Figure 4: HackBench paper figure" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Figure 4. **Recall drops on harder hack subsets.** Arrows run from easier to harder; leftward means recall falls. Columns: hack origin (natural → synthetic) and scope (general → task-specific). Synthetic hacks are generally harder to catch than natural hacks in single pass, but this pattern reverses under chunking. Task-specific hacks, on the other hand, are generally harder to catch in both settings._

**Which hacks are missed?** Figure 4 breaks recall down by hack origin (natural vs. synthetic) and scope (general vs. task-specific; see Table 1). Across 13 models and two monitoring scaffolds, recall is lower on the prespecified harder subset in 38 of 52 comparisons.

<details markdown="1">
<summary>See breakdown by hack origin and scope</summary>

Synthetic hacks are harder to detect than natural hacks in 17 of 26 comparisons. Under single-pass monitoring, synthetic-hack recall is lower for 11 of 13 models, with a median gap of -9.1 percentage points. Under chunking, however, synthetic-hack recall is lower for only 6 of 13 models, and the median difference reverses to +4.5 points. This pattern is consistent with a retrieval failure in single-pass monitoring: synthetic hacks are localized insertions in otherwise legitimate transcripts, and chunking makes the relevant span easier to find.

Hack type behaves differently. Under single-pass monitoring, task-specific recall is lower for 12 of 13 models, with a median gap of -11.1 points. Under chunking, task-specific recall remains lower for 9 of 13 models, and the median gap widens to -16.5 points.

</details>

### Scaling Monitoring Compute

In this section, we investigate three axes of scaling monitor compute across three models (Kimi K3, GLM 5.2, and Claude Opus 4.8): reasoning effort, the number of chunks, and the number of tool calls in an agentic setting where the transcript is placed in a file system and the model can use bash tools to inspect it.

**Scaling Reasoning Effort and Number of Chunks**

We report how binary detection F₁ on HackBench scales along two axes of test-time compute: reasoning effort and the number of chunks per transcript (Figure 5). More reasoning effort has little and inconsistent impact on hack detection (left). Opus peaks at 0.621 with high effort, compared with 0.581 at low and 0.584 at max. Kimi peaks at 0.681 with high effort, compared with 0.679 at none and 0.652 at max. GLM peaks at 0.320 with minimal effort, declines to 0.223 at high, and recovers to 0.315 at max. Chunking produces much larger gains (right). From 1 to 16 chunks, F₁ increases from 0.315 to 0.797 for GLM-5.2 and from 0.584 to 0.783 for Claude Opus 4.8. Kimi K3 increases from 0.652 to a peak of 0.829 at two chunks and remains above 0.80 through 16 chunks.

<a id="fig-scaling_effort_reasoning_chunks"></a>

<div class="hackbench-figure" style="--figure-width: 90%;">
  <a href="fig4.png"><img src="fig4.png" alt="Figure 5: HackBench paper figure" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Figure 5. Binary detection F1 as a function of reasoning effort (left) and the number of chunks {1, 2, 4, ..., 16} (right) for Claude Opus 4.8, GLM 5.2, and Kimi K3. Increasing reasoning effort provides little and inconsistent improvement, whereas chunking the transcript produces substantially larger gains._

**Agentic Scaling.**

<a id="fig-agentic-scaling"></a>

<div class="hackbench-figure" style="--figure-width: 89%;">
  <a href="fig5.png"><img src="fig5.png" alt="Figure 6: HackBench paper figure" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Figure 6. Binary detection F₁ versus cost per question (log scale, USD, prefix-cached) for Claude Opus 4.8, Kimi K3, and GLM 5.2 monitors with *agentic scaling* on HackBench. The monitor is given a file system holding the transcript and bash tools to search and read it._

Rather than placing the transcript in context, we give the monitor a filesystem holding the transcript and tools to search and read it. This lifts two constraints of single-pass monitoring: the transcript no longer has to fit in the monitor's context window, and the monitor pays only for what it reads instead of prefilling every token. Figure 6 reports estimated cost (USD per question) against binary detection F₁.

At eight tool calls, agentic monitoring costs 10.4× less than single pass for Kimi K3, 17.6× less for Opus 4.8, and 11.9× less for GLM-5.2. Kimi reaches F₁=0.634 at 64 calls but remains below single pass (0.683), while costing 1.7× less. Opus reaches 0.613 at 32 calls, exceeding single pass (0.584), while costing 5.6× less. GLM already exceeds single pass at eight calls (0.427 versus 0.333) while costing 11.9× less. These results show that agentic scaffolds can outperform single-pass monitoring at a fraction of the estimated cost for some models. **They further suggest that, as models' agentic monitoring capabilities improve, agentic scaffolds could provide better oversight than single-pass monitoring while remaining far cheaper.** However, the current gains remain model-dependent and do not increase monotonically with the tool-call budget.

## Hack Localization Results

In this section, we evaluate harder variants of HackBench beyond binary prediction (hack or legitimate). We also ask the monitor to localize which exact command(s) are executing the hack. We present two variants: **numbered localization**, where each tool call is labeled with a number and the model must output the exact set of numbers, and **verbatim localization**, where the model must emit the actual line(s) of code within a tool call that are hacking.

<details markdown="1">
<summary>See localization metrics</summary>

**Setup** We report the following three scores for models with 1M context window on hack localization:

- **Located-any** (localization recall): the fraction of hacking transcripts for which the monitor recovers *at least one* gold item---one of the hacking commands (verbatim) or indices (numbered). It measures whether the monitor can point to the hack at all, and is insensitive to over-flagging.
- **Exact-match** (strict localization): the fraction of hacking transcripts for which the predicted set of commands/indices is *exactly* the gold set. This is the strictest criterion, penalizing both misses and over-inclusion.
- **Jaccard** (set level similarity): let P be the monitor's predicted set of hack commands and G the ground-truth set of hack commands. Jaccard is defined as (\|P ∩ G\|) / (\|P ∪ G\|), the set-level similarity between P and G.

</details>

<div class="hackbench-figure" style="--figure-width: 85%; --mobile-width: 900px;">
  <a href="table3.png"><img src="table3.png" alt="Table 3: results" /></a>
</div>

<p class="hackbench-figure-hint">Swipe to explore the figure; tap to open it at full size.</p>

_Table 3. Single-pass command-localization performance on HackBench. All localization metrics are reported as percentages over HACK trajectories. Jaccard denotes the mean similarity between predicted and gold command sets. Avg. EM and Avg. Jaccard are the unweighted means across the verbatim and numbered localization formats. Results are sorted by Avg. Jaccard._

**Results and Findings** Table 3 reports both variants across models. Recovering the complete set of hacking commands remains challenging: the best exact-match score is 20.25% for verbatim localization and 45.57% for numbered localization, so no model exactly recovers the gold command set on even half of the HACK examples. Located-any scores are substantially higher than exact-match scores, indicating that monitors often identify at least one hacking command but fail to recover the complete set. Numbered localization yields higher located-any, exact-match, and Jaccard scores for every model shown, suggesting that selecting command indices is easier and less error-prone than reproducing commands verbatim.



<details markdown="1">
<summary>See related work</summary>

## Related Work

**Reward hacking and specification gaming.** Reward hacking---high reward without the intended behavior---is a long-standing concern in reinforcement learning ([Amodei et al., 2016][amodei2016concrete]; [Krakovna et al., 2020][krakovna2020specification]; [Skalse et al., 2022][skalse2022defining]; [Pan et al., 2022][pan2022effects]; [Manheim et al., 2018][manheim2018categorizing]). In the RLHF/RLVR era it reappears as reward-model overoptimization ([Gao et al., 2022][gao2022scaling]; [Moskovitz et al., 2023][moskovitz2023confronting]; [Eisenstein et al., 2024][eisenstein2024helping]) and, more sharply, as *reward tampering*, where a model edits its own reward channel ([Denison et al., 2024][denison2024reward]). We study reward hacking in a setting where the “reward” is a concrete automated verifier the agent can read and modify, and where hacks are multi-step programs rather than single actions.

**Autonomous research agents.** Agentic coding and tool use ([Yao et al., 2022][yao2022react]; [Schick et al., 2023][schick2023toolformer]; [Jimenez et al., 2024][jimenez2024swebench]; [Yang et al., 2024][yang2024sweagent]) now extend to end-to-end ML research, and a wave of benchmarks measures it: MLE-bench ([Chan et al., 2024][chan2024mlebench]), RE-Bench ([Wijk et al., 2024][wijk2024rebench]), PaperBench ([Starace et al., 2025][starace2025paperbench]), MLAgentBench ([Huang et al., 2024][huang2024mlagentbench]), ResearchGym ([Garikaparthi et al., 2026][garikaparthi2026researchgymevaluatinglanguagemodel]), MLS-Bench ([Lyu et al., 2026][lyu2026mlsbenchholisticrigorousassessment]), and AutoLab ([Xu et al., 2026][autolab-2026]). These works optimize for and measure *capability*. We instead treat their tasks as a testbed in which to study *integrity*: the same open-ended, verifier-scored task that measures research skill also invites reward hacking the verifier. Closely related to our work, [Huang et al. (2026)][huang2026rewardhackingchallengesoversight] studies when reward hacking occurs and what forms it takes. Our work complements [Huang et al. (2026)][huang2026rewardhackingchallengesoversight] by studying how well language-model-based monitors detect and locate reward hacking in autoresearch transcripts.

**Oversight, monitoring, and control.** Detecting misbehavior from a model's behavior or reasoning is central to scalable oversight ([Bowman et al., 2022][bowman2022measuring]), weak-to-strong generalization ([Burns et al., 2023][burns2023weaktostrong]), and AI control ([Greenblatt et al., 2024][greenblatt2024control]). Chain-of-thought monitoring can catch reward hacking but is fragile if optimized against ([Baker et al., 2025][baker2025monitoring]), and reasoning traces are not always faithful ([Turpin et al., 2023][turpin2023language]); frontier models can scheme and hide intent when incentivized ([Meinke et al., 2024][meinke2024scheming]; [Hubinger et al., 2024][hubinger2024sleeper]). To monitor agents, [Kutasov et al. (2025)][kutasov2025shadearenaevaluatingsabotagemonitoring] propose asking agents to perform both a main task and a covert side task in general agentic settings (e.g. booking an airplane ticket), then checking whether monitors catch the side task. [Libon et al. (2026)][libon2026researcharenaevaluatingsabotagemonitoring] extends the main/side task red teaming protocol to autoresearch settings. However, [Martin et al. (2026)][martin2026classifiercontextrotmonitor] and [Jotautaitė et al. (2026)][jotautaitė2026monitoringbenchsemiautomatedredteamingagent] find that existing monitoring benchmarks may overestimate monitor performance because they evaluate monitors on relatively short trajectories and a narrow range of hacks. In our work, we mitigate these limitations by injection red-teaming --- directly asking a model to inject different hack commands into long agent transcripts.

**LLM judges and long context.** We build on LLM-as-a-judge methods ([Zheng et al., 2023][zheng2023judging]; [Li et al., 2023][alpaca_eval]; [Dubois et al., 2024][dubois2024lcwr]), using LLMs to assess whether agent trajectories contain reward hacking. Because these trajectories can be very long, monitoring is subject to known failure modes such as lost-in-the-middle effects and context-window limits ([Liu et al., 2023][liu2023lost]; [Hsieh et al., 2024][hsieh2024ruler]). We therefore apply test-time scaling ([Wang et al., 2022][wang2022self]; [Snell et al., 2024][snell2024scalingllmtesttimecompute]): we either divide each trajectory into chunks and aggregate the resulting judgments, or allow an agentic monitor to search the transcript. Relatedly, [Kale et al. (2025)][kale2025reliableweaktostrongmonitoringllm] and [sfereido (2026)][sfereido2026chunked] show that chunking can improve recall when monitoring for misaligned behavior.


</details>


## Conclusion

Autoresearch agents reward-hack because we ask them to maximize metrics, and they do so without proper regard to other implicit constraints, i.e. performing actions humans would consider acceptable. We thus built HackBench, a benchmark that measures whether hacks inside autoresearch trajectories can be detected, and showed that current frontier models do not fully succeed at this task. As agents take on more of the research loop, monitoring reward hacking is not an add-on but a prerequisite; to support future work, we [release trajectories](https://huggingface.co/datasets/facebook/hackbench) containing naturally occurring and synthetically injected hacks, together with human-audited command-level gold labels.


## Contributors
Tianjian Li, Seungone Kim, Jason Weston, Ilia Kulikov, Jack Lanchantin, Weizhe Yuan

## More details
We plan to put a full technical report on arXiv soon.

## Citation
You can cite this blog (before the full paper is released) here:

```bibtex
@article{li2026hackbench,
  title   = "HackBench: Detecting and Localizing Reward Hacking in Autoresearch Agents",
  author  = {Li, Tianjian and Kim, Seungone and Weston, Jason and Kulikov, Ilia and Lanchantin, Jack and Yuan, Weizhe},
  year    = "2026",
  month   = "October",
  url     = "https://facebookresearch.github.io/RAM/blogs/hackbench/"
}
```


<!-- Links for the citations above; identifiers follow the original LaTeX citation keys. -->
[chan2024mlebench]: https://arxiv.org/abs/2410.07095
[wijk2024rebench]: https://arxiv.org/abs/2411.15114
[starace2025paperbench]: https://arxiv.org/abs/2504.01848
[huang2024mlagentbench]: https://arxiv.org/abs/2310.03302
[skalse2022defining]: https://arxiv.org/abs/2209.13085
[pan2022effects]: https://arxiv.org/abs/2201.03544
[weng2024rewardhack]: https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
[posttrainbench2026v11]: https://posttrainbench.com/blog/posttrainbench-1-1/
[modded_nanogpt_2024]: https://github.com/KellerJordan/modded-nanogpt
[metr2026expenditure]: https://metr.org/blog/2026-07-21-expenditure-horizon/
[deepseekai2025deepseekr1incentivizingreasoningcapability]: https://arxiv.org/abs/2501.12948
[lightman2024lets]: https://openreview.net/forum?id=v8L0pN6EOi
[anthropic2026rewardseeker]: https://alignment.anthropic.com/2026/reward-seeker/
[liu2023lost]: https://arxiv.org/abs/2307.03172
[hsieh2024ruler]: https://arxiv.org/abs/2404.06654
[baker2025monitoring]: https://arxiv.org/abs/2503.11926
[garikaparthi2026researchgymevaluatinglanguagemodel]: https://arxiv.org/abs/2602.15112
[autolab-2026]: https://arxiv.org/abs/2606.05080
[lyu2026mlsbenchholisticrigorousassessment]: https://arxiv.org/abs/2605.08678
[karpathy2026autoresearch]: https://github.com/karpathy/autoresearch
[safetyresearch2026automatedw2s]: https://github.com/safety-research/automated-w2s-research
[Harbor_Framework]: https://doi.org/10.5281/zenodo.20953922
[openai2026gpt55]: https://deploymentsafety.openai.com/gpt-5-5
[zai2026glm52]: https://z.ai/blog/glm-5.2
[amodei2016concrete]: https://arxiv.org/abs/1606.06565
[krakovna2020specification]: https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
[manheim2018categorizing]: https://arxiv.org/abs/1803.04585
[gao2022scaling]: https://arxiv.org/abs/2210.10760
[moskovitz2023confronting]: https://arxiv.org/abs/2310.04373
[eisenstein2024helping]: https://openreview.net/forum?id=5u1GpUkKtG
[denison2024reward]: https://arxiv.org/abs/2406.10162
[yao2022react]: https://arxiv.org/abs/2210.03629
[schick2023toolformer]: https://arxiv.org/abs/2302.04761
[jimenez2024swebench]: https://arxiv.org/abs/2310.06770
[yang2024sweagent]: https://arxiv.org/abs/2405.15793
[huang2026rewardhackingchallengesoversight]: https://arxiv.org/abs/2609.28614
[bowman2022measuring]: https://arxiv.org/abs/2211.03540
[burns2023weaktostrong]: https://arxiv.org/abs/2312.09390
[greenblatt2024control]: https://arxiv.org/abs/2312.06942
[turpin2023language]: https://arxiv.org/abs/2305.04388
[meinke2024scheming]: https://arxiv.org/abs/2412.04984
[hubinger2024sleeper]: https://arxiv.org/abs/2401.05566
[kutasov2025shadearenaevaluatingsabotagemonitoring]: https://arxiv.org/abs/2506.15740
[libon2026researcharenaevaluatingsabotagemonitoring]: https://arxiv.org/abs/2607.19321
[martin2026classifiercontextrotmonitor]: https://arxiv.org/abs/2605.12366
[jotautaitė2026monitoringbenchsemiautomatedredteamingagent]: https://arxiv.org/abs/2605.09684
[zheng2023judging]: https://arxiv.org/abs/2306.05685
[alpaca_eval]: https://github.com/tatsu-lab/alpaca_eval
[dubois2024lcwr]: https://arxiv.org/abs/2404.04475
[wang2022self]: https://arxiv.org/abs/2203.11171
[snell2024scalingllmtesttimecompute]: https://arxiv.org/abs/2408.03314
[kale2025reliableweaktostrongmonitoringllm]: https://arxiv.org/abs/2508.19461
[sfereido2026chunked]: https://www.lesswrong.com/posts/orMEvTgtRig9qnsdL/using-chunked-monitoring-to-detect-deception-in-long
