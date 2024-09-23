# Self-Taught Evaluators
TODO: insert a figure.

## Inference and Evaluation
Coming soon.

## Synthetic Preference Data
### Generate worse response
1. Given pairs of (instruction, response), run generation using the template specified in `data/prompts/worse_response.prompt`.
2. Run generation on the prompts from step 1, with temperature 0.7, and top_p=0.9
### Generate judgement

## Model Training
Coming soon.
## Citation
If you use data, model, or code from this work, please cite with the following BibTex entry:

@article{wang2024self,
  title={Self-taught evaluators},
  author={Wang, Tianlu and Kulikov, Ilia and Golovneva, Olga and Yu, Ping and Yuan, Weizhe and Dwivedi-Yu, Jane and Pang, Richard Yuanzhe and Fazel-Zarandi, Maryam and Weston, Jason and Li, Xian},
  journal={arXiv preprint arXiv:2408.02666},
  year={2024}
}
