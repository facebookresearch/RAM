"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Tuple

from graph import Edge, Node
from rules import (
    CheckRule,
    InferenceRule,
    SymmetryInference,
    TriangleInference,
    UniqueRule,
)


class IsAtEdge(Edge):
    type = "is_at"
    text = "is at"
    sta_type = ["person", "item"]
    end_type = "place"
    can_be_question = True
    question_on_sta = True

    @classmethod
    def is_obs_sta_candidate(cls, node: Node) -> bool:
        if not super().is_obs_sta_candidate(node):
            return False
        # item location is not directly observed
        return node.type == "person"

    @classmethod
    def make_question(cls, node: Node) -> str:
        return f"Q: Where is {node.name}?"


class IsWithEdge(Edge):
    type = "is_with"
    text = "is with"
    sta_type = "person"
    end_type = "person"
    question_on_sta = True

    @classmethod
    def make_question(cls, node: Node) -> str:
        return f"Q: Who is with {node.name}?"


class HasEdge(Edge):
    type = "has"
    text = "has"
    sta_type = "person"
    end_type = "item"
    can_be_question = True
    question_on_sta = False

    @classmethod
    def make_question(cls, node: Node) -> str:
        return f"Q: Who has {node.name}?"


class IsInsideEdge(Edge):
    type = "is_inside"
    text = "is inside"
    sta_type = "item"
    end_type = "item"

    @classmethod
    def is_sta_candidate(cls, node: Node) -> bool:
        if not super().is_sta_candidate(node):
            return False
        return "is_small" in node.properties

    @classmethod
    def is_end_candidate(cls, node: Node) -> bool:
        if not super().is_end_candidate(node):
            return False
        return "is_container" in node.properties


def create_rules() -> Tuple[List[CheckRule], List[InferenceRule]]:
    check_rules = []
    inference_rules = []

    # [is_at] relation
    # cannot be at two different places
    check_rules.append(UniqueRule(IsAtEdge))

    # [has] relation
    # item cannot belong to two different people
    check_rules.append(UniqueRule(HasEdge, for_end_node=True))
    # A has B + A is at C => B is at C
    inference_rules.append(TriangleInference(HasEdge, IsAtEdge, IsAtEdge))
    # B has A + A is at C => B is at C
    inference_rules.append(
        TriangleInference(HasEdge, IsAtEdge, IsAtEdge, reverseAB=True)
    )

    # [is_with] relation
    # let's limit to 2 people for now
    check_rules.append(UniqueRule(IsWithEdge))
    # A is with B => B is with A
    inference_rules.append(SymmetryInference(IsWithEdge))
    # A is with B + A is with C => B is with C
    inference_rules.append(TriangleInference(IsWithEdge, IsWithEdge, IsWithEdge))
    # A is with B + A is at C => B is at C
    inference_rules.append(TriangleInference(IsWithEdge, IsAtEdge, IsAtEdge))
    # B is at A + C is at A => B is with C
    inference_rules.append(
        TriangleInference(
            IsAtEdge,
            IsAtEdge,
            IsWithEdge,
            typeB="person",
            typeC="person",
            reverseAB=True,
            reverseAC=True,
        )
    )

    # [is_inside] relation
    # can't be inside two different things
    check_rules.append(UniqueRule(IsInsideEdge))
    # B has A + C is inside A => B has C
    inference_rules.append(
        TriangleInference(
            HasEdge, IsInsideEdge, HasEdge, reverseAB=True, reverseAC=True
        )
    )
    # B has A + A is inside C => B has C (e.g. Alice has key + key is inside bag => Alice has bag)
    inference_rules.append(
        TriangleInference(HasEdge, IsInsideEdge, HasEdge, reverseAB=True)
    )

    # # near relation
    # check_rules.append(UniqueRule("near"))  # this might not be that realistic
    # inference_rules.append(SymmetryInference("near", typeA="place", typeB="place"))

    return check_rules, inference_rules
