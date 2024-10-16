"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional, Type

from graph import Edge, Graph


class CheckRule:
    """Rules that must be always true"""

    def check(self, graph: Graph) -> bool:
        raise NotImplementedError


class InferenceRule:
    """If certain conditions met, create a new relation"""

    def infer(self, graph: Graph) -> Optional[Edge]:
        """return the edge if a new relation is created"""
        raise NotImplementedError


class UniqueRule(CheckRule):
    """Given relation AB should be unique for node A"""

    def __init__(
        self,
        rel_class: Type[Edge],
        for_end_node=False,
        typeA: Optional[str] = None,
        typeB: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.rel_class = rel_class
        self.for_end_node = for_end_node
        self.typeA = typeA
        self.typeB = typeB

    def check(self, graph: Graph) -> bool:
        edges = graph.get_edges(self.rel_class.type)
        for edge in edges:
            nodeA, nodeB = edge.node_sta, edge.node_end
            if self.typeA is not None and nodeA.type != self.typeA:
                continue
            if self.typeB is not None and nodeB.type != self.typeB:
                continue
            if not self.for_end_node:
                edgeAB = nodeA.get_edges(self.rel_class.type)
            else:
                edgeAB = nodeB.get_edges(self.rel_class.type, True)
            if len(edgeAB) > 1:
                return False
        return True


class SymmetryInference(InferenceRule):
    """AB relation should match BA relation"""

    def __init__(
        self,
        rel_class: Type[Edge],
        typeA: Optional[str] = None,
        typeB: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.rel_class = rel_class
        self.typeA = typeA
        self.typeB = typeB

    def infer(self, graph: Graph):
        edges = graph.get_edges(self.rel_class.type)
        for edge in edges:
            nodeA, nodeB = edge.node_sta, edge.node_end
            if self.typeA is not None and nodeA.type != self.typeA:
                continue
            if self.typeB is not None and nodeB.type != self.typeB:
                continue
            if not graph.exist_edge(nodeB, self.rel_class.type, nodeA):
                new_edge = graph.add_edge(nodeB, self.rel_class, nodeA, supports=[edge])
                return new_edge
        return None


class TriangleInference(InferenceRule):
    """Given relations between AB and AC, create a new relation BC"""

    def __init__(
        self,
        relAB: Type[Edge],
        relAC: Type[Edge],
        relBC: Type[Edge],
        typeA: Optional[str] = None,
        typeB: Optional[str] = None,
        typeC: Optional[str] = None,
        reverseAB=False,
        reverseAC=False,
        reverseBC=False,
    ) -> None:
        super().__init__()
        self.relAB = relAB
        self.relAC = relAC
        self.relBC = relBC
        self.typeA = typeA
        self.typeB = typeB
        self.typeC = typeC
        self.reverseAB = reverseAB
        self.reverseAC = reverseAC
        self.reverseBC = reverseBC

    def infer(self, graph: Graph):
        triangles = graph.find_triangle(
            self.relAB.type, self.relAC.type, self.reverseAB, self.reverseAC
        )
        for nodeA, nodeB, nodeC, edgeAB, edgeAC in triangles:
            if self.typeA is not None and nodeA.type != self.typeA:
                continue
            if self.typeB is not None and nodeB.type != self.typeB:
                continue
            if self.typeC is not None and nodeC.type != self.typeC:
                continue

            # check if BC relation already exists
            if not self.reverseBC:
                if graph.exist_edge(nodeB, self.relBC.type, nodeC):
                    continue
            else:
                if graph.exist_edge(nodeC, self.relBC.type, nodeB):
                    continue

            # create BC relation
            supports = [edgeAB, edgeAC]
            if not self.reverseBC:
                new_edge = graph.add_edge(nodeB, self.relBC, nodeC, supports=supports)
            else:
                new_edge = graph.add_edge(nodeC, self.relBC, nodeB, supports=supports)
            return new_edge
        return None
