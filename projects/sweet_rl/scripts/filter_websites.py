"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC-By-NC license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from fire import Fire
import numpy as np
from tqdm import tqdm
from sweet_rl.utils.webpage_utils import get_driver, render_html


def main(
    saved_path="/fsx-ram/yifeizhou/collab_llm/outputs/code_collab_train.jsonl",
    filtered_path=None,
    temp_path="/fsx-ram/yifeizhou/collab_llm/driver_cache",
):
    filtered_trajectories = []
    driver = get_driver()
    with open(saved_path, "r") as fb:
        raw_data = [json.loads(line) for line in fb]
    for d in tqdm(raw_data):
        if not "ground_truth" in d or not "problem_description" in d:
            continue
        if render_html(driver, d["ground_truth"], temp_path, 0) is not None:
            filtered_trajectories.append(d)

    print(f"Number of raw trajectories: {len(raw_data)}")
    print(f"Number of filtered trajectories: {len(filtered_trajectories)}")
    with open(filtered_path, "w") as fb:
        for trajectory in filtered_trajectories:
            fb.write(json.dumps(trajectory) + "\n")
    print(f"Filtered data saved to {filtered_path}")


if __name__ == "__main__":
    Fire(main)
