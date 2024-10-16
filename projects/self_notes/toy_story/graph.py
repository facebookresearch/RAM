"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Dict, List, Type, Union


class Node:
    def __init__(self, name: str, type: str, properties: List[str] = []) -> None:
        self.name = name
        self.type = type
        self.properties = properties
        self._edges_out: Dict[str, List[Edge]] = {}
        self._edges_in: Dict[str, List[Edge]] = {}

    def connect(self, edge: "Edge", is_reverse: bool = False) -> None:
        edges = self._edges_in if is_reverse else self._edges_out
        if edge.type not in edges:
            edges[edge.type] = []
        edges[edge.type].append(edge)

    def disconnect(self, edge: "Edge", is_reverse: bool = False) -> None:
        edges = self._edges_in if is_reverse else self._edges_out
        edges[edge.type].remove(edge)
        if len(edges[edge.type]) == 0:
            del edges[edge.type]

    def get_edges(self, rel_type: str, is_reverse: bool = False) -> List["Edge"]:
        edges = self._edges_in if is_reverse else self._edges_out
        if rel_type not in edges:
            return []
        return edges[rel_type]

    def has_rel(self, rel_type: str, is_reverse: bool = False) -> bool:
        edges = self.get_edges(rel_type, is_reverse)
        return len(edges) > 0

    def get_connected_nodes(
        self, rel_type: str, is_reverse: bool = False
    ) -> List["Node"]:
        nodes: List[Node] = []
        for edge in self.get_edges(rel_type, is_reverse):
            if is_reverse:
                nodes.append(edge.node_sta)
            else:
                nodes.append(edge.node_end)
        return nodes

    def __str__(self) -> str:
        return self.name


class Edge:
    type: str
    text: str
    sta_type: Union[str, List[str]]
    end_type: Union[str, List[str]]
    can_be_question = False
    question_on_sta = True  # the question about node_sta, not node_end

    def __init__(
        self, node_sta: Node, node_end: Node, supports: List["Edge"] = []
    ) -> None:
        self.node_sta = node_sta
        self.node_end = node_end
        self.node_sta.connect(self)
        self.node_end.connect(self, is_reverse=True)

        # if created by inference based on other relations
        self.supports = supports  # supporting relations. usually 2
        self.support_obss: List[Edge] = []  # supporting observations
        for edge in self.supports:
            if edge.is_observed:
                self.support_obss.append(edge)
            else:
                # support must contain only observed relations
                self.support_obss.extend(edge.support_obss)

    @classmethod
    def is_sta_candidate(cls, node: Node) -> bool:
        """ " Is the node a valid candidate"""
        if cls.sta_type is None:
            return True
        elif type(cls.sta_type) is list:
            return node.type in cls.sta_type
        else:
            return node.type == cls.sta_type

    @classmethod
    def is_obs_sta_candidate(cls, node: Node) -> bool:
        """ " Is the node a valid candidate for observation"""
        return cls.is_sta_candidate(node)

    @classmethod
    def is_end_candidate(cls, node: Node) -> bool:
        """ " Is the node a valid candidate"""
        if cls.end_type is None:
            return True
        elif type(cls.end_type) is list:
            return node.type in cls.end_type
        else:
            return node.type == cls.end_type

    @classmethod
    def is_obs_end_candidate(cls, node: Node) -> bool:
        """ " Is the node a valid candidate for observation"""
        return cls.is_end_candidate(node)

    def disconnect(self) -> None:
        self.node_sta.disconnect(self)
        self.node_end.disconnect(self, True)

    @property
    def is_observed(self) -> bool:
        # should have no support if it is directly observed
        return len(self.supports) == 0

    @property
    def is_simple(self) -> bool:
        # directly observed or can be inferred from a single observation
        return len(self.support_obss) < 2

    def to_text(self) -> str:
        """convert to natural language"""
        return f"{self.node_sta} {self.text} {self.node_end}."

    @classmethod
    def make_question(cls, node: Node) -> str:
        raise NotImplementedError

    def to_question(self) -> str:
        if self.question_on_sta:
            return self.make_question(self.node_sta)
        else:
            return self.make_question(self.node_end)

    def __str__(self) -> str:
        s = f"{self.node_sta} {self.type} {self.node_end}."
        if self.is_observed:
            s += " [obs]"
        return s


class Graph:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self._node_by_name: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._edges_by_rel: Dict[str, List[Edge]] = {}
        self._nodes_by_type: Dict[str, List[Node]] = {}

    @property
    def edges(self) -> List[Edge]:
        return self._edges.copy()

    def add_node(self, name: str, type: str, properties: List[str] = []) -> None:
        assert name not in self.nodes
        node = Node(name, type, properties=properties)
        self.nodes.append(node)
        assert name not in self._node_by_name
        self._node_by_name[name] = node
        if type not in self._nodes_by_type:
            self._nodes_by_type[type] = []
        self._nodes_by_type[type].append(node)

    def get_node(self, name: str) -> Node:
        return self._node_by_name[name]

    def get_nodes_by_type(self, type: str) -> List[Node]:
        return self._nodes_by_type[type].copy()

    def add_edge(
        self,
        node_sta: Node,
        rel_class: Type[Edge],
        node_end: Node,
        supports: List[Edge] = [],
    ) -> Edge:
        edge = rel_class(node_sta, node_end, supports=supports)
        self._edges.append(edge)
        if edge.type not in self._edges_by_rel:
            self._edges_by_rel[edge.type] = []
        self._edges_by_rel[edge.type].append(edge)
        return edge

    def remove_edge(self, edge: Edge) -> None:
        edge.disconnect()
        self._edges.remove(edge)
        self._edges_by_rel[edge.type].remove(edge)

    def get_edges(self, rel_type: str) -> List[Edge]:
        if rel_type not in self._edges_by_rel:
            return []
        return self._edges_by_rel[rel_type]

    def __str__(self) -> str:
        str = ""
        for edge in self.edges:
            str += f"{edge}\n"
        return str

    def exist_edge(self, node_sta: Node, rel_type: str, node_end: Node) -> bool:
        """Check if a given relation exist between nodes exists"""
        edges = node_sta.get_edges(rel_type)
        for edge in edges:
            if node_end == edge.node_end:
                return True
        return False

    def find_triangle(
        self, relAB: str, relAC: str, reverseAB: bool = False, reverseAC: bool = False
    ) -> List[Any]:
        """Find nodes A, B, C with given relations"""
        triangles: List[Any] = []

        edgesAB = self.get_edges(relAB)
        for edgeAB in edgesAB:
            if reverseAB:
                nodeB, nodeA = edgeAB.node_sta, edgeAB.node_end
            else:
                nodeA, nodeB = edgeAB.node_sta, edgeAB.node_end
            edgesAC = nodeA.get_edges(relAC, reverseAC)
            for edgeAC in edgesAC:
                if reverseAC:
                    nodeC = edgeAC.node_sta
                else:
                    nodeC = edgeAC.node_end
                if nodeA is nodeC or nodeB is nodeC:
                    continue
                triangles.append([nodeA, nodeB, nodeC, edgeAB, edgeAC])
        return triangles
