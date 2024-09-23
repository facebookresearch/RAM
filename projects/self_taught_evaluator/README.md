# Self-Taught Evaluators

<p align="center"><img width="90%" src="figures/self_taught_sft.png" /></p>

## Inference and Evaluation
Coming soon.

## Synthetic Preference Data
### Generate worse response
1. Given pairs of (instruction, baseline response), prepare prompts using the template specified in `data/prompts/worse_response.prompt`.
2. Run generation on the prompts from step 1, to generate a "worse response" to the instruction. 
### Generate judgement 
1. Given tuples of (instruction, baseline response, worse response), prepare prompts using the template specified in `data/prompts/eval_plan.prompt`.
2. Run generation on the prompts from step 1 to derive evaluation plans for pairwise preference. Then we apply rejection sampling, where we collect multiple samples of evaluation plan, and only retain examples where the judgement prefers the baseline response to the worse response. 

The experiments in the paper used sampling with temperature=0.7, and top_p=0.9.
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
