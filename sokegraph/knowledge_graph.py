"""
knowledge_graph.py

Defines an abstract base class for all knowledge‑graph builders used in
SOKEGraph (e.g., Neo4jGraph, NetworkXGraph, etc.).  Subclasses must
implement `build_graph`, which takes the loaded ontology extractions and
creates the actual graph in the chosen backend.
"""

from abc import ABC, abstractmethod
import json
from typing import Any


class KnowledgeGraph(ABC):
    """Abstract base class for building a knowledge graph from an ontology.

    Attributes
    ----------
    ontology_path : str
        Path to the ontology‑extraction JSON file.
    ontology_extractions : dict
        Parsed JSON loaded from `ontology_path`, ready for graph construction.
    """

    # ------------------------------------------------------------------ #
    # Constructor & helpers                                              #
    # ------------------------------------------------------------------ #
    def __init__(self, ontology_path: str) -> None:
        """
        Load ontology extractions from disk and store them for later use.

        Parameters
        ----------
        ontology_path : str
            Path to the JSON file produced by the extraction pipeline.
        """
        self.ontology_path = ontology_path
        self.ontology_extractions = self._load_ontology()

    def _load_ontology(self) -> Any:
        """Read the ontology JSON into a Python object."""
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    # Abstract API every concrete graph builder must implement           #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_graph(self) -> None:
        """Convert ``self.ontology_extractions`` into a graph structure.

        Concrete subclasses (e.g. `Neo4jKnowledgeGraph`) should implement the
        entire graph‑creation workflow here.
        """
        pass

    @abstractmethod
    def show_graph(self):
        pass


    @abstractmethod
    def get_attr_keys(self):
        """Return all attribute names that appear on any node
        (e.g. ['layer', 'keyword', 'domain', ...])."""

    @abstractmethod
    def get_attr_values(self, key: str):
        """Return distinct values for *key* (e.g. all layer names)."""

    @abstractmethod
    def subgraph_for_attr(self, key: str, value: str):
        """Return sub‑graph containing only nodes with attr[key]==value."""

    @abstractmethod
    def neighbour_subgraph(self, node_id: str) :
        """Return node + first‑degree neighbours (for drill‑down)."""
