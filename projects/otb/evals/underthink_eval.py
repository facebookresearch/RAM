"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import re

import reasoning_gym


def _extract_boxed(text: str, find_last_box: bool = False) -> str:
    try:
        if find_last_box:
            return text.split("\\boxed{")[-1].split("}")[0]
        return text.split("\\boxed{")[1].split("}")[0]
    except Exception:
        return text


def _conditional_format_answer(response: str, puzzle: str) -> str:
    if puzzle == "ab":
        response = re.sub(r"[^#AB]", " ", response).replace("  ", " ")
    elif puzzle == "mini_sudoku":
        response = response.replace("\\\\", "\n").replace("\\", " ")
    response = response.replace("\\ ", " ").replace("\\ ", " ")
    return response


def eval_underthink(row, find_last_box: bool = False) -> float:
    puzzle = json.loads(row["metadata"])["source_dataset"]
    responses = (
        row["response"] if isinstance(row["response"], list) else [row["response"]]
    )
    entry = row.to_dict() if hasattr(row, "to_dict") else row
    data = reasoning_gym.create_dataset(
        json.loads(entry["metadata"])["source_dataset"], size=1
    )
    scores = []
    for res in responses:
        res = _extract_boxed(res, find_last_box=find_last_box)
        res = _conditional_format_answer(res if res is not None else "", puzzle)
        entry_for_grade = json.loads(entry["metadata"])
        entry_for_grade["puzzle"] = puzzle
        entry_for_grade["answer"] = entry["answer"]
        entry_for_grade["metadata"] = json.loads(entry["metadata"])
        scores.append(data.score_answer(answer=res, entry=entry_for_grade))
    return sum(scores) / len(scores) if len(scores) > 0 else 0.0
