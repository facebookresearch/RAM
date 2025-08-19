"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import pandas as pd
import reasoning_gym
from reasoning_gym.factory import create_dataset

puzzles = {
    "ab": {"length": 20},
    "advanced_geometry": {"min_coord": -20, "max_coord": 20},
    "knight_swap": {},
    "tsumego": {"min_board_size": 15, "max_board_size": 19, "max_stones": 20},
    "fraction_simplification": {},
    "propositional_logic": {},
    "bitwise_arithmetic": {"difficulty": 4},
    "letter_counting": {"min_words": 25, "max_words": 35},
    "maze": {"min_dist": 15, "max_dist": 25, "min_grid_size": 15, "max_grid_size": 25},
    "puzzle24": {"min_value": 8, "max_value": 10},
    "quantum_lock": {"difficulty": 25},
}

all_data = []
for puzzle in puzzles:
    data = reasoning_gym.create_dataset(puzzle, size=50, **puzzles[puzzle])
    data = list(data)
    for x in data:
        x["puzzle"] = puzzle
    all_data.extend(data)


df = pd.DataFrame(all_data)
os.makedirs("data", exist_ok=True)
df.to_pickle("data/underthink_data.pkl")
