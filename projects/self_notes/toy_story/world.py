"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from constants import (
    CONTAINER_ITEMS,
    MAX_NUM_SUPPORT,
    MAX_OBS_TRIAL,
    MAX_QUESTION_TRIAL,
    PEOPLE,
    PLACES,
    QUESTION_RELATIONS,
    RELATIONS,
    SMALL_ITEMS,
)
from graph import Edge, Graph
from relation import HasEdge, IsAtEdge, IsInsideEdge, IsWithEdge, create_rules
from rules import TriangleInference


class World:
    """Toy story world class"""

    def __init__(self) -> None:
        self.check_rules, self.inference_rules = create_rules()
        self.graph = Graph()

        for name in PEOPLE:
            self.graph.add_node(name, "person")
        for name in SMALL_ITEMS:
            self.graph.add_node(
                name, "item", properties=["is_small"]
            )  # TODO: surely there is a better way
        for name in CONTAINER_ITEMS:
            self.graph.add_node(name, "item", properties=["is_container"])
        for name in PLACES:
            self.graph.add_node(name, "place")

    def generate_story(
        self,
        num_obs: int,
        include_inference_qa: str,
        add_support: bool,
        qa_prob: float,
    ) -> Tuple[List[str], Optional[List[str]]]:
        """Generate story"""

        story: List[str] = []
        scratchpad = [] if include_inference_qa == "after" else None

        if include_inference_qa == "mixed":
            add_question = True
            if qa_prob < 1.0:
                # For training where certain samples have intermediate questions (IQs) while certain samples don't
                # we denote the ones which don't have IQs by this prefix token "NQ"
                if np.random.random() > qa_prob:
                    story.append("NQ")  # No question to be added
                    add_question = False

        for _ in range(num_obs):
            obs, inferred_edges = None, []
            for _ in range(MAX_OBS_TRIAL):
                obs, inferred_edges, _ = self.try_generate_obs()
                if obs is not None:
                    break
            if obs is None:
                # couldn't generate a new obs
                break

            story.append(obs.to_text())

            if include_inference_qa == "mixed" and add_question:
                for e in inferred_edges:
                    story.append("S" + e.to_question())
                    if add_support:
                        for se in e.supports:
                            if se.to_text() != obs.to_text():
                                # Only copy facts which are not the previous observation itself
                                story.append(se.to_text())
                    story.append(e.to_text())

            elif include_inference_qa == "after":
                for e in inferred_edges:
                    if add_support:
                        for se in e.supports:
                            scratchpad.append(se.to_text())
                    scratchpad.append(e.to_question())
                    scratchpad.append(e.to_text())

        assert len(story) > 0
        return story, scratchpad

    def try_generate_obs(
        self, obs_rel_class: Optional[Type[Edge]] = None
    ) -> Tuple[Optional[Edge], Optional[List[Edge]], Optional[List[Edge]]]:
        if obs_rel_class is None:
            obs_rel_class = random.choice(RELATIONS)
        candidates_sta = [
            n for n in self.graph.nodes if obs_rel_class.is_obs_sta_candidate(n)
        ]
        candidates_end = [
            n for n in self.graph.nodes if obs_rel_class.is_obs_end_candidate(n)
        ]

        node_sta = random.choice(candidates_sta)
        if node_sta in candidates_end:
            candidates_end.remove(node_sta)
        node_end = random.choice(candidates_end)

        # check if the relation already exists
        if self.graph.exist_edge(node_sta, obs_rel_class.type, node_end):
            return None, None, None

        obs_edge = self.graph.add_edge(node_sta, obs_rel_class, node_end)
        new_edges = [obs_edge]
        inferred_edges = []

        # run all inference rules
        run_inference = True
        while run_inference:
            run_inference = False
            for rule in self.inference_rules:
                new_edge = rule.infer(self.graph)
                if new_edge is not None:
                    new_edges.append(new_edge)
                    # record only 2-hop edges
                    if isinstance(rule, TriangleInference):
                        inferred_edges.append(new_edge)
                    # restart each time an edge is created
                    run_inference = True
                    break

        for rule in self.check_rules:
            if not rule.check(self.graph):
                # check failed, reverse all changes
                for edge in new_edges:
                    self.graph.remove_edge(edge)
                logging.debug(f"rejected = {str(obs_edge)}")
                return None, None, None

        for e in new_edges:
            logging.debug(f"new edge = {e}")

        return obs_edge, inferred_edges, new_edges

    def generate_question(
        self,
        filter_nsupports: Optional[List[int]] = None,
        balance_nsupports=False,
        unknown=0.0,
    ):
        if unknown > 0:
            if random.random() < unknown:
                return self.generate_unknown_question()

        nsupports_target = None
        if balance_nsupports:
            if filter_nsupports is None:
                nsupports_target = random.randint(1, MAX_NUM_SUPPORT)
            else:
                nsupports_target = random.choice(filter_nsupports)

        for _ in range(MAX_QUESTION_TRIAL):
            rel_class = random.choice(QUESTION_RELATIONS)
            edges = self.graph.get_edges(rel_class.type)
            if len(edges) == 0:
                continue
            edge = random.choice(edges)
            nsupports = len(edge.support_obss)
            if nsupports == 0:
                nsupports = 1  # no difference in 0 and 1 support? TODO: fix it
            assert nsupports <= MAX_NUM_SUPPORT
            if balance_nsupports:
                if nsupports != nsupports_target:
                    continue
            elif filter_nsupports is not None:
                if nsupports not in filter_nsupports:
                    continue

            qa_dict = self.edge2question(edge)
            return qa_dict
        return None

    def edge2question(self, edge: Edge) -> Dict[str, Any]:
        question = edge.to_question()
        answer = edge.to_text()

        supports: List[str] = []
        sub_questions: List[str] = []
        if not edge.is_observed:
            supports = [s.to_text() for s in edge.support_obss]
            assert len(edge.supports) == 2

            if not edge.supports[0].is_simple:
                # can't be inferred easily, so need to ask question
                subq = self.edge2question(edge.supports[0])
                sub_questions.append(subq["question"])

            if not edge.supports[1].is_simple:
                subq = self.edge2question(edge.supports[1])
                sub_questions.append(subq["question"])

        return {
            "question": question,
            "answer": answer,
            "supports": supports,
            "sub_questions": sub_questions,
        }

    def generate_unknown_question(self):
        """Function to intentionally generate unanswerable questions"""
        for _ in range(MAX_QUESTION_TRIAL):
            # Using numpy random here so that python's random module is not affected by
            # without random choices being made here (to avoid random module's state change)
            rel_class = np.random.choice(QUESTION_RELATIONS)

            if rel_class.question_on_sta:
                candidates = [
                    n for n in self.graph.nodes if rel_class.is_sta_candidate(n)
                ]
            else:
                candidates = [
                    n for n in self.graph.nodes if rel_class.is_end_candidate(n)
                ]

            node = random.choice(candidates)

            # check if the relation already exists
            if node.has_rel(rel_class.type, is_reverse=not rel_class.question_on_sta):
                continue

            qa = {
                "question": rel_class.make_question(node),
                "answer": "unknown",
                "supports": [],
                "sub_questions": [],
            }

            return qa
        return None
