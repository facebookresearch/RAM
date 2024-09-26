
## Projects
Here we list projects undertaken in the RAM framework that are shared publicly, either in the form of papers, public tasks and/or shared model code. This directory also contains subfolders for some of the projects which are housed in the RAM repo, others are maintained via external websites.

### [Following Length Constraints in Instructions](https://arxiv.org/abs/2406.17744)
Aligned instruction following models can better fulfill user requests than their unaligned counterparts. However, it has been shown that there is a length bias in evaluation of such models, and that training algorithms tend to exploit this bias by learning longer responses. In this work we show how to train models that can be controlled at inference time with instructions containing desired length constraints. Such models are superior in length instructed evaluations, outperforming standard instruction following models such as GPT4, Llama 3 and Mixtral.

### [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/pdf/2405.18719)
The attention mechanism is a critical component of Large Language Models (LLMs) that allows tokens in a sequence to interact with each other but is orderinvariant. Incorporating position encoding (PE) makes it possible to address by position, such as attending to the i-th token. However, current PE methods use token counts to derive position, and thus cannot generalize to higher levels of abstraction, such as attending to the i-th sentence. In this paper, we propose a new position encoding method, Contextual Position Encoding (CoPE), that allows positions to be conditioned on context by incrementing position only on certain tokens determined by the model. This allows more general position addressing such as attending to the i-th particular word, noun, or sentence. We show that CoPE can solve the selective copy, counting and Flip-Flop tasks where popular position embeddings fail, and improves perplexity on language modeling and coding tasks.

### [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666)
Model-based evaluation is at the heart of successful model development -- as a reward model for training, and as a replacement for human evaluation. To train such evaluators, the standard approach is to collect a large amount of human preference judgments over model responses, which is costly and the data becomes stale as models improve. In this work, we present an approach that aims to im-prove evaluators without human annotations, using synthetic training data only. Starting from unlabeled instructions, our iterative self-improvement scheme generates contrasting model outputs and trains an LLM-as-a-Judge to produce reasoning traces and final judgments, repeating this training at each new iteration using the improved predictions. Without any labeled preference data, our Self-Taught Evaluator can improve a strong LLM (Llama3-70B-Instruct) from 75.4 to 88.3 (88.7 with majority vote) on RewardBench. This outperforms commonly used LLM judges such as GPT-4 and matches the performance of the top-performing reward models trained with labeled examples.

<!---
### [System 2 Attention (is something you might need too)](https://arxiv.org/pdf/2311.11829.pdf)
Soft attention in Transformer-based Large Language Models (LLMs) is susceptible to incorporating irrelevant information from the context into its latent representations, which adversely affects next token generations. To help rectify these issues, we introduce System 2 Attention (S2A), which leverages the ability of LLMs to reason in natural language and follow instructions in order to decide what to attend to. S2A regenerates the input context to only include the relevant portions, before attending to the regenerated context to elicit the final response. In experiments, S2A outperforms standard attention-based LLMs on three tasks containing opinion or irrelevant information, QA, math word problems and longform generation, where S2A increases factuality and objectivity, and decreases sycophancy.


### [Some things are more CRINGE than others: Preference Optimization with the Pairwise Cringe Loss](https://arxiv.org/pdf/2312.16682.pdf)
Practitioners commonly align large language models using pairwise preferences, i.e., given labels of the type response A is preferred to response B for a given input. Perhaps less commonly, methods have also been developed for binary feedback, i.e. training models given labels of type response A is good or bad. We show how an existing performant binary feedback method, the Cringe Loss (Adolphs et al., 2022), can be generalized to the pairwise preference setting using a simple soft margin extension. Pairwise Cringe Loss is straightforward to implement and efficient to train, and we find it outperforms state-of-the-art preference optimization algorithms such as PPO and DPO on the AlpacaFarm benchmark.


## Data
The data needed to run our code is hosted on HuggingFace:
- https://huggingface.co/OpenAssistant
- https://huggingface.co/datasets/tatsu-lab/alpaca_eval

## Model
The library needed to run our code is
- [Llama from HuggignFace] (https://huggingface.co/docs/transformers/main/model_doc/llama?fbclid=IwAR2ZRhVnuKqngWTBjhOhuDgQLQ5yzTh573uAA_16bEMX3lerKSHCtdla31w).To run huggingface Llama models, make sure to convert your LLaMA checkpoint and tokenizer into HuggingFace format and store it at <your_path_to_hf_converted_llama_ckpt_and_tokenizer>.
- [Alpaca Eval](https://github.com/tatsu-lab/alpaca_eval) for any inference only Llama experiments.
-->
