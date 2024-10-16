"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Type

from graph import Edge
from relation import HasEdge, IsAtEdge, IsInsideEdge, IsWithEdge, create_rules

PEOPLE = ["Alice", "Bob", "Charlie", "Daniel", "Frank", "Mary", "John", "Sandra"]
PLACES = [
    "the farm",
    "the park",
    "the museum",
    "the lake",
    "the bridge",
    "the station",
    "the town hall",
    "the store",
]
SMALL_ITEMS = ["the key", "the apple", "the banana", "the ball", "the book"]
CONTAINER_ITEMS = ["the bag", "the box", "the suitcase", "the basket"]
ITEMS = SMALL_ITEMS + CONTAINER_ITEMS

RELATIONS: List[Type[Edge]] = [IsAtEdge, IsWithEdge, HasEdge, IsInsideEdge]
QUESTION_RELATIONS = [r for r in RELATIONS if r.can_be_question]
MAX_NUM_SUPPORT = 5
MAX_OBS_TRIAL = 25
MAX_QUESTION_TRIAL = 50
MAX_SAMPLE_TRIAL = 100
