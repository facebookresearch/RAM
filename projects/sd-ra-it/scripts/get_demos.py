"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import itertools as it
import json
import logging
import operator as op
import os
import pathlib
import random
import re
from glob import glob
from typing import Iterable

import fire  # type:ignore
import numpy as np
from src.utils import Answer, Dataset, OfflineModel, batched, get_renderer
from tqdm import tqdm

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Strat:
    def __call__(self, dataset: Dataset) -> list[Answer]:
        raise NotImplementedError

    def parse(self, summary):
        match = re.match(r"Here is|Here's", summary.strip(), flags=re.DOTALL)
        if match:
            summary = ":".join(summary.split(":")[1:]).strip()
        else:
            summary = summary.strip()
        logger.info(summary)
        return summary


class OptimizedNoRag(Strat):
    def __init__(self, model, sys, prompt_id):
        self.model = model
        self.sys = sys
        self.prompt_id = prompt_id
        self.summarize = pathlib.Path("templates/summarize.jinja").read_text()

    def __call__(self, dataset):
        usr = [line["question"] for line in dataset]
        sys = list(it.repeat(self.sys, len(usr)))
        summarize = list(it.repeat(self.summarize, len(usr)))
        outputs = self.model.series(sys, usr, summarize)
        return [self.parse(output[0]) for output in outputs]

    def __repr__(self):
        return "NoRag{}".format(self.prompt_id)


class OptimizedRag(Strat):
    def __init__(self, model, sys, prompt_id):
        self.model = model
        self.sys = sys
        self.prompt_id = prompt_id
        self.render = get_renderer("templates/iterative/generate/usr.jinja")
        self.summarize = pathlib.Path("templates/summarize.jinja").read_text()

    def __call__(self, dataset):
        usr = [
            self.render(question=line["question"], contexts=line["retrieved_support"])
            for line in dataset
        ]
        sys = list(it.repeat(self.sys, len(usr)))
        summarize = list(it.repeat(self.summarize, len(usr)))
        outputs = self.model.series(sys, usr, summarize)
        return [self.parse(output[0]) for output in outputs]

    def __repr__(self):
        return "Rag{}".format(self.prompt_id)


def choose_best(
    model,
    sys,
    dataset,
    preds_lists: list[list[str]],  # Strats, Lines
) -> tuple[list[str], list[Strat]]:
    render = get_renderer("templates/judge/usr.jinja")
    usr = [
        render(
            question=line["question"],
            answer=line["answer"],
            predictions=[pred for pred in preds if pred is not None],
            contexts=line["retrieved_support"][:4],
        )
        for line, preds in zip(dataset, preds_lists)
    ]
    parsed_choices = [None for _ in usr]
    for attempt in range(5):
        parsed_is_none = [parsed is None for parsed in parsed_choices]
        unfinished = list(it.compress(usr, parsed_is_none))
        if not unfinished:
            break
        choices = model.zero_shot(sys=it.repeat(sys, len(unfinished)), usr=unfinished)
        choices_iter = iter(choices)
        parsed_choices = [
            (
                choice
                if choice is not None
                else parse(next(choices_iter), len(preds_lists[0]))
            )
            for choice in parsed_choices
        ]
    predictions: list[str | None] = [
        (choice if choice is None else preds[choice])
        for choice, preds in zip(parsed_choices, preds_lists)
    ]
    return predictions


def tournament(model, dataset, predictions_list, strategies, sys, bracket_size=3):
    strats = list(map(str, strategies))
    strat_lookups = [
        dict(zip(line_preds, strats)) for line_preds in zip(*predictions_list)
    ]
    preds = np.array(predictions_list)  # Strats, Lines
    preds_shuf = rng.permuted(preds, axis=0)  # Strats, Lines
    while len(preds_shuf) > bracket_size:
        winner_preds_list = []
        for bracket in batched(preds_shuf, bracket_size):  # Shape: Strats_subset, Lines
            bracket_preds = np.array(bracket).T  # Shape: Lines, Strats_subset
            winner_preds = choose_best(model, sys, dataset, bracket_preds)
            winner_preds_list.append(winner_preds)
        preds_shuf = np.array(winner_preds_list)  # Shape: Brackets, Lines
    winner_preds = choose_best(model, sys, dataset, preds_shuf.T)  # Lines
    non_none_total = sum(pred is not None for pred in winner_preds)
    logger.info(f"Non-None predictions: {non_none_total}")
    winner_strats = list(map(dict.get, strat_lookups, winner_preds))
    return winner_preds, winner_strats


def parse(reply: str, max_idx: int):
    match = re.search(r"Best:(?: ?Prediction) (?P<idx>\d)", reply)
    if match:
        idx = int(match["idx"])
        if 1 <= idx <= max_idx:
            return idx - 1


def process_batch(
    dataset,
    strategies,
    model,
):
    logger.info("Processing new batch")
    sys: str = pathlib.Path("templates/judge/sys.jinja").read_text()
    predictions_list = [strat(dataset) for strat in strategies]
    predictions, winners = tournament(model, dataset, predictions_list, strategies, sys)
    dataset_with_preds = [
        line | dict(prediction=prediction, prediction_source=winner, predictions=preds)
        for line, prediction, winner, preds in zip(
            dataset, predictions, winners, zip(*predictions_list)
        )
        if prediction is not None
    ]
    logger.info(f"Processed examples: {len(dataset_with_preds)}")
    return dataset_with_preds


def get_strats(model, cls, prompts):
    return [cls(model, prompt["prompt"], i) for i, prompt in enumerate(prompts)]


def main(
    filename: str = "ra-dit/multisource/oasst.jsonl",
    output_file: str = "data/demos.jsonl",
    n: int | None = None,
    prompts_per_strat: int = 6,
    batch_size: int = 5000,
    continued: bool = False,
    logfile: str = "logs/get_demos.log",
    tensor_parallel_size: int | None = None,
    model_name: str = "Meta-Llama-3-8B-Instruct",
):
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode="w")
    model_params = dict(tensor_parallel_size=tensor_parallel_size)
    model = OfflineModel(
        model_name,
        max_tokens=1000,
        temperature=1.0,
        model_params=model_params,
    )
    with open("data/v0/prompts/norag.jsonl") as file:
        no_rag_prompts = list(map(json.loads, it.islice(file, prompts_per_strat)))
    with open("data/v0/prompts/rag.jsonl") as file:
        rag_prompts = list(map(json.loads, it.islice(file, prompts_per_strat)))
    no_rag_strats = get_strats(model, OptimizedNoRag, no_rag_prompts)
    rag_strats = get_strats(model, OptimizedRag, rag_prompts)
    refuse_prompt = pathlib.Path("templates/refuse.jinja").read_text()
    # no_rag_strats.append(OptimizedNoRag(model, refuse_prompt, "Refuse"))
    rag_strats.append(OptimizedRag(model, refuse_prompt, "Refuse"))
    rag_strats.append(OptimizedRag(model, refuse_prompt, "Refuse"))
    rag_strats.append(OptimizedRag(model, refuse_prompt, "Refuse"))
    strategies = no_rag_strats + rag_strats
    if continued and os.path.exists(output_file):
        with open(output_file) as file:
            starting_point = len(list(file))
    else:
        starting_point = 0
    with open(output_file, "a" if continued else "w") as outfile, open(
        filename
    ) as infile:
        whole_dataset = map(json.loads, infile)
        dataset = it.islice(whole_dataset, starting_point, n)
        output_lines = it.chain.from_iterable(
            process_batch(batch, strategies, model)
            for batch in batched(dataset, batch_size)
        )
        output_line_slice = it.islice(output_lines, n - starting_point)
        for line in tqdm(
            output_line_slice,
            desc="Saving demos",
            total=n,
            initial=starting_point,
        ):
            print(json.dumps(line), file=outfile, flush=True)


if __name__ == "__main__":
    fire.Fire(main)
