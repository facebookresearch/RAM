"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import functools as ft
import itertools as it
import json
import logging
import pathlib
import random
import re
import statistics

import fire  # type:ignore
from src.utils import Dataset, Ex, OfflineBaseModel, OfflineModel, Pmpt, get_renderer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_scores(model, dataset, predictions: Pmpt[Ex[str]]) -> Pmpt[Ex[float]]:
    render_example = get_renderer("templates/iterative/score/example.jinja")
    sys_prompt = pathlib.Path("templates/iterative/score/sys.jinja").read_text()
    user_prompts: Pmpt[Ex[str]] = []
    for prediction_list in predictions:
        contexts, questions, answers = get_questions_and_answers(dataset)
        prompts = [
            render_example(question=q, answer=a, prediction=p)
            for q, a, p in zip(questions, answers, prediction_list)
        ]
        user_prompts.append(prompts)
    sys = it.repeat(sys_prompt)
    unparsed_scores: Pmpt[Ex[str]] = [
        model.zero_shot(sys=sys, usr=usr)
        for usr in tqdm(user_prompts, desc="Scoring prompts")
    ]
    scores: Pmpt[Ex[float]] = list(map(parse, unparsed_scores))
    return scores


def get_top_k(scores, *args, k=None, reverse=True):
    assert (
        len(set(map(len, (scores, *args)))) == 1
    ), f"Lengths not equal: {list(map(len, (scores, *args)))}"
    shuffled = random.sample(list(zip(scores, *args)), k=len(scores))
    sorted_by_score = sorted(shuffled, reverse=reverse)
    top_k = list(sorted_by_score)[:k]
    scores, *args = zip(*top_k)
    return scores, *args


def rewrite(
    model,
    prompts: Pmpt[str],
    dataset: Dataset,
    predictions: Pmpt[Ex[str]],
    scores: Pmpt[Ex[float]],
    samples=None,
    rag=False,
) -> list[str]:
    assert (
        len(prompts) == len(scores) == len(predictions)
    ), f"Bad shapes for scores ({len(prompts)=}, {len(scores)=}, {len(predictions)=})"
    rewrite_sys: str = pathlib.Path("templates/iterative/rewrite/sys.jinja").read_text()
    if rag:
        render_critique = get_renderer("templates/iterative/rewrite/rag_critique.jinja")
    else:
        render_critique = get_renderer("templates/iterative/rewrite/critique.jinja")
    render_rewrite = get_renderer("templates/iterative/rewrite/rewrite.jinja")
    critique_prompts: Pmpt[Ex[str]] = []
    rewrite_prompts: Pmpt[str] = []
    for prompt, preds, pred_scores in zip(prompts, predictions, scores):
        contexts, questions, answers = get_questions_and_answers(dataset)
        pred_scores, preds, questions, answers = get_top_k(
            pred_scores, preds, questions, answers, k=samples
        )
        critique_prompts.append(
            [
                render_critique(question=q, answer=a, prediction=p, contexts=c)
                for q, a, p, c in zip(questions, answers, preds, contexts)
            ]
        )
        rewrite_prompts.append(render_rewrite(prompt=prompt))
    critique_prompts_by_example: Ex[Pmpt[str]] = list(map(list, zip(*critique_prompts)))
    rewrites = model.series(
        list(it.repeat(rewrite_sys, len(rewrite_prompts))),
        *critique_prompts_by_example,
        rewrite_prompts,
        final_sampling_params=dict(n=samples, temperature=1.0),
    )
    assert (
        len(rewrites[0]) == samples
    ), f"Fewer ({len(rewrites[0])}) rewrites than expected ({samples})"
    assert len(rewrites) == len(
        prompts
    ), f"Fewer ({len(rewrites)}) rewrite lists than expected ({len(prompts)})"
    flattened_rewrites = [
        parse_system_prompt(rewritten_prompt)
        for rewrite_list in rewrites
        for rewritten_prompt in rewrite_list
    ]
    assert len(flattened_rewrites) == samples * len(
        prompts
    ), f"Not enough rewrites ({len(flattened_rewrites)})"
    return flattened_rewrites


def parse_system_prompt(prompt):
    if ":" in prompt and re.match(r"Here is|Here's", prompt):
        return ":".join(prompt.split(":")[1:]).strip().strip('"')
    else:
        return prompt


def generate(model, prompts, dataset, rag=False):
    predictions = []
    contexts, questions, _ = get_questions_and_answers(dataset)
    if rag:
        render = get_renderer("templates/iterative/generate/usr.jinja")
        usr = [render(question=q, contexts=c) for q, c in zip(questions, contexts)]
    else:
        usr = questions
    for prompt in tqdm(prompts, desc="Predicting responses"):
        outputs = model.zero_shot(sys=list(it.repeat(prompt, len(questions))), usr=usr)
        predictions.append(outputs)
    return predictions


def parse(score_strs: list[str]) -> list[float]:
    pattern = re.compile(r"Score: (?P<score>\d)")
    matches = map(pattern.search, score_strs)
    scores = []
    for match in matches:
        if match is None:
            scores.append(0.0)
        else:
            scores.append(float(int(match["score"])))
    return scores


def get_questions_and_answers(dataset):
    tuples = [
        (line["retrieved_support"], line["question"], line["answer"])
        for line in dataset
    ]
    return (*zip(*tuples),)


def main(
    dataset_filename: str = "ra-dit/multisource/oasst.jsonl",
    model: str = "Meta-Llama-3-8B-Instruct/",
    outfile: str = "templates/iterative/optimized.json",
    logfile: str = "iterative.log",
    eval_example_count: int = 100,
    train_example_count: int = 100,
    topk: int = 4,
    shuffle_window: int = 400,
    beam_size: int = 12,
    rag: bool = False,
    chat: bool = True,
    tensor_parallel_size=1,
    steps: int = 10,
):
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode="w")
    starting_prompt = pathlib.Path("templates/no_rag/system.jinja").read_text()
    prompts = [starting_prompt]
    model_params = dict(tensor_parallel_size=tensor_parallel_size)
    if chat:
        llm = OfflineModel(
            model, max_tokens=1000, temperature=0.5, model_params=model_params
        )
    else:
        llm = OfflineBaseModel(
            model, max_tokens=1000, temperature=0.1, model_params=model_params
        )
    random.seed(0)
    with open(dataset_filename) as file:
        dataset: Dataset = random.sample(
            list(map(json.loads, it.islice(file, shuffle_window))),
            k=eval_example_count + train_example_count,
        )
    train_set, eval_set = dataset[:train_example_count], dataset[-eval_example_count:]
    all_prompts = []
    steps = steps
    for step in range(steps):
        predictions: list[list[str]] = generate(llm, prompts, train_set, rag=rag)
        pred_scores: list[list[float]] = get_scores(llm, train_set, predictions)
        prompt_scores: list[float] = list(map(statistics.mean, pred_scores))  # str
        logger.info(f"str scores: {prompt_scores}")
        topk_prompt_scores, topk_prompts, topk_scores, topk_preds = get_top_k(
            prompt_scores, prompts, pred_scores, predictions, k=topk
        )
        all_prompts.extend(prompts)
        prompts_count: int = len(prompts)
        prompts = rewrite(
            llm,
            topk_prompts,
            train_set,
            topk_preds,
            topk_scores,
            samples=beam_size // topk,
            rag=rag,
        )
    val_preds = generate(llm, all_prompts, eval_set, rag=rag)
    val_pred_scores = get_scores(llm, eval_set, val_preds)
    val_prompt_scores = list(map(statistics.mean, val_pred_scores))
    with open(outfile, "w") as file:
        scores, prompts = get_top_k(val_prompt_scores, all_prompts, k=len(all_prompts))
        for score, prompt in zip(scores, prompts):
            print(json.dumps(dict(prompt=prompt, score=score)), file=file)


if __name__ == "__main__":
    fire.Fire(main)
