from abc import ABC, abstractmethod

class KnowledgeGraph(ABC):
    """Abstract base class for building knowledge graphs."""

    def __init__(self, ontology_path: str):
        self.ontology_path = ontology_path
        self.ontology_extractions = self._load_ontology()

    def _load_ontology(self):
        import json
        with open(self.ontology_path, "r") as f:
            return json.load(f)

    @abstractmethod
    def build_graph(self):
        pass
