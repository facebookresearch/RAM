# Self-Taught Evaluators

<p align="center"><img width="90%" src="figures/self_taught_dpo.png" /></p>

## Inference and Evaluation
Coming soon.

## Synthetic Preference Data
### Generate worse response
1. Given pairs of (instruction, baseline response), prepare prompts using the template specified in `data/prompts/worse_response.prompt`.
2. Run generation on the prompts from step 1, to generate a "worse response" to the instruction. 
### Generate judgement 
1. Given tuples of (instruction, baseline response, worse response), we generate judgement using the prompt template specified in `data/prompts/eval_plan.prompt`. To avoid position bias, we generate evaluation plans for both orders of the responses positions. Specifically, for `0_1` order, we prepare the prompt using (instruction, baseline response, worse response), and for `1_0` order, we prepare the prompt using (instruction, worse response, baseline response).
2. Run generation on both `0_1` and `1_0` ordered prompts from step 1 to derive evaluation plans for pairwise preference.
3. Then we apply rejection sampling, where we collect multiple samples of evaluation plan, and only retain examples where the judgement prefers the baseline response to the worse response. To ensure label balance, we retain the same number of examples of `A is better` and `B is better`.

### vllm Sampling
The experiments in the paper used vllm for generation, with temperature=0.7, and top_p=0.9, max_tokens=4096.

## Model Training
### SFT
Coming soon.
### DPO
Coming soon.
## Citation
If you use data, model, or code from this work, please cite with the following BibTex entry:
```
@article{wang2024self,
  title={Self-taught evaluators},
  author={Wang, Tianlu and Kulikov, Ilia and Golovneva, Olga and Yu, Ping and Yuan, Weizhe and Dwivedi-Yu, Jane and Pang, Richard Yuanzhe and Fazel-Zarandi, Maryam and Weston, Jason and Li, Xian},
  journal={arXiv preprint arXiv:2408.02666},
  year={2024}
}
```
