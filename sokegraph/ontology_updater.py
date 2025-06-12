import json
from sokegraph.ai_agent import AIAgent
import os

from sokegraph.functions import safe_title, parse_all_metadata

class OntologyUpdater:
    def __init__(self, ontology_path, papers, ai_tool: AIAgent, output_dir: str):
        self.ontology_path = ontology_path
        self.papers = papers
        self.ai_tool = ai_tool
        self.output_dir = output_dir
        self.ontology = None
        self.load_ontology()
        self.output_path = None

    def load_ontology(self):
        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)

    def save_ontology(self, output_path):
        if self.ontology is None:
            raise ValueError("Ontology not updated. Nothing to save.")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.ontology, f, indent=2, ensure_ascii=False)
    
    def enrich_with_papers(self) -> str:
        """
        Enrich ontology using paper abstracts and an AI agent.
        """

        text_data = self._get_text_data()
        self.ontology = self.ai_tool.extract_keywords(self.ontology, text_data)
        parse_all_metadata(self.ontology)

        self.output_path = f"{self.output_dir}/updated_ontology.json"
        self.save_ontology(self.output_path)
        return self.ontology
    
    def _get_text_data(self):
        text_data = {}
        for paper in self.papers:
            safe_id = safe_title(paper['title'] or paper['paper_id'])
            text_data[safe_id] = paper['abstract'] or ""
        return text_data