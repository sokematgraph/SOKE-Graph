"""
ontology_updater.py

Provides OntologyUpdater — a helper that enriches an ontology JSON file
using AI‑extracted keywords from a list of papers, then persists the updated
ontology to disk.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict

from sokegraph.agents.ai_agent import AIAgent
from sokegraph.utils.functions import safe_title, parse_all_metadata


class OntologyUpdater:
    """
    Enriches an ontology with new keywords/metadata extracted from papers.

    Workflow
    --------
    1. Load existing ontology JSON.
    2. Use an :class:`sokegraph.ai_agent.AIAgent` to extract keywords from
       each paper abstract.
    3. Merge the new keywords / parsed metadata back into the ontology.
    4. Save the result to ``output_dir/updated_ontology.json``.

    Parameters
    ----------
    ontology_path : str
        Path to the base ontology JSON file.
    papers : list[dict]
        Paper metadata list; each dict should have ``title``, ``paper_id``,
        and ``abstract`` keys.
    ai_tool : AIAgent
        Concrete AI agent (OpenAI, Gemini, etc.) used for extraction.
    output_dir : str
        Directory where the updated ontology will be written.
    """

    # ------------------------------------------------------------------ #
    # Construction / I/O helpers                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        ontology_path: str,
        papers: List[Dict],
        ai_tool: AIAgent,
        output_dir: str,
    ):
        self.ontology_path = ontology_path
        self.papers = papers
        self.ai_tool = ai_tool
        self.output_dir = output_dir

        self.ontology: dict | None = None
        self.output_path: str | None = None

        self._load_ontology()

    # Public: enrich & save -------------------------------------------- #
    def enrich_with_papers(self) -> str:
        """Enrich the ontology and return the output file path.

        Returns
        -------
        str
            Path to the saved, updated ontology JSON file.
        """
        # Build {safe_paper_id: abstract_text} for AI extraction
        text_data = self._build_text_data()

        # Ask the AI agent to extract keywords & merge into ontology
        self.ontology = self.ai_tool.extract_keywords(self.ontology, text_data)

        # Parse numeric metadata strings (overpotential, Tafel slope, etc.)
        parse_all_metadata(self.ontology)

        # Persist to disk
        self.output_path = os.path.join(self.output_dir, "updated_ontology.json")
        self._save_ontology(self.output_path)

        return self.output_path

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _load_ontology(self) -> None:
        """Load ontology JSON from ``self.ontology_path`` into memory."""
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

    def _save_ontology(self, output_path: str) -> None:
        """Save the in‑memory ontology to disk."""
        if self.ontology is None:
            raise ValueError("Ontology not updated. Nothing to save.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.ontology, f, indent=2, ensure_ascii=False)

    def _build_text_data(self) -> Dict[str, str]:
        """
        Convert paper list into ``{safe_id: abstract}`` mapping.

        Returns
        -------
        dict
            Keys are sanitized titles (or paper IDs), values are abstracts.
        """
        text_data: Dict[str, str] = {}
        for paper in self.papers:
            safe_id = safe_title(paper.get("title") or paper["paper_id"])
            text_data[safe_id] = paper.get("abstract", "")
        return text_data