"""
ai_agent.py

Abstract base class for all AI agents used in SOKEGraph.
It defines a common interface (`ask`) plus shared utilities for:

* Loading API keys
* Building prompts for structured information extraction
* Parsing LLM output (JSON first, regex fallback)
* Deriving synonyms / opposites for keywords
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import json
import logging
import re
from typing import Dict, List

LOG = logging.getLogger(__name__)


class AIAgent(ABC):
    """Abstract base class for any large‑language‑model agent.

    Sub‑classes (e.g. `OpenAIAgent`, `GeminiAgent`, `LlamaAgent`) must
    implement :py:meth:`ask`, which sends a prompt to the underlying model
    and returns the raw text response.
    """

    # ------------------------------------------------------------------ #
    # API surface that *must* be implemented by concrete subclasses
    # ------------------------------------------------------------------ #
    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Send a prompt to the model and return the textual response."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared utility methods available to all child classes
    # ------------------------------------------------------------------ #
    # 1) Key management ------------------------------------------------- #
    @staticmethod
    def load_api_keys(api_file_path: str) -> List[str]:
        """Load one key per line from a plaintext file.

        Args:
            api_file_path: Path to a file containing API keys.

        Returns:
            A list of non‑empty keys stripped of whitespace.
        """
        with open(api_file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # 2) Prompt construction ------------------------------------------- #
    def _build_prompt(self, layer_name: str,
                      abstract_text: str,
                      ontology_layer: dict) -> str:
        """Create a structured extraction prompt for a single ontology layer.

        Args:
            layer_name: Name of the ontology layer (e.g. “Reaction”).
            abstract_text: Abstract text from the research paper.
            ontology_layer: Dict of categories → sample keywords.

        Returns:
            Formatted prompt string.
        """
        return f"""
                    You are a structured information extraction AI specialized in materials science.

                    You will extract keywords relevant to the layer: **{layer_name}**
                    For this layer, the following categories and sample keywords are provided:
                    {json.dumps(ontology_layer, indent=2)}

                    From the text below, extract:
                    - Any new or related keywords
                    - The category it fits under
                    - Any nearby numeric values with units (e.g., overpotential, pH, current density)

                    Text to analyze:
                    {abstract_text}

                    Return the output in this exact JSON list format:
                    [
                    {{
                        "layer": "LayerName",
                        "matched_category": "Category",
                        "keyword": "ExtractedKeyword",
                        "meta_data": "Text snippet"
                    }}
                    ]
                """

    # 3) High‑level keyword extraction --------------------------------- #
    def extract_keywords(self,
                         ontology: dict,
                         text_data: Dict[str, str]) -> dict:
        """Run the extraction pipeline across many papers.

        Args:
            ontology: Full ontology dict {layer: {category: [keywords]}}.
            text_data: Mapping {paper_id: abstract_text}.

        Returns:
            Nested dict results[layer][category] -> List[dict] with keys:
              keywords, meta_data, paper_id
        """
        results_dict = defaultdict(lambda: defaultdict(list))

        for paper_id, abstract in text_data.items():
            LOG.info("🔍 Processing paper: %s", paper_id)
            for layer, categories in ontology.items():
                # Call model & parse
                results = self._call_model(layer, abstract, categories)

                # Merge results back into ontology structure
                for res in results:
                    cat = res["matched_category"]
                    if cat not in ontology[layer]:
                        LOG.warning("⚠️ Unknown category '%s' under layer '%s'",
                                    cat, layer)
                        continue

                    # Deduplicate keywords
                    keywords = list(set(ontology[layer][cat] + [res["keyword"]]))
                    results_dict[layer][cat].append(
                        dict(keywords=keywords,
                             meta_data=res["meta_data"],
                             paper_id=paper_id)
                    )

                LOG.info("✅ Extracted %d items for paper %s layer '%s'",
                         len(results), paper_id, layer)
        return results_dict

    # 4) Private helpers ------------------------------------------------ #
    def _call_model(self, layer: str,
                    abstract_text: str,
                    ontology_layer: dict) -> List[dict]:
        """Utility that builds the prompt, queries the model and parses."""
        prompt = self._build_prompt(layer, abstract_text, ontology_layer)
        print("xxxx")
        model_output = self.ask(prompt)
        return [] if not model_output else self._parse_model_output(model_output)

    @staticmethod
    def _parse_model_output(model_output: str) -> List[dict]:
        """Parse model output—JSON first, regex fallback.

        Args:
            model_output: Raw string response from the model.

        Returns:
            A list of dictionaries (possibly empty) with expected keys.
        """
        # First try strict JSON
        try:
            parsed = json.loads(model_output)
            LOG.info("✅ JSON parsed successfully (%d items).", len(parsed))
            return parsed
        except json.JSONDecodeError:
            LOG.warning("⚠️ JSON parsing failed. Falling back to regex.")

            # Very loose pattern; adjust if schema changes
            pattern = (r'"layer":\s*"([^"]+)",\s*"matched_category":\s*"([^"]+)",'
                       r'\s*"keyword":\s*"([^"]+)",\s*"meta_data":\s*"([^"]+)"')
            extracted = [
                dict(layer=m[0].strip(),
                     matched_category=m[1].strip(),
                     keyword=m[2].strip(),
                     meta_data=m[3].strip())
                for m in re.findall(pattern, model_output, re.DOTALL)
            ]
            LOG.info("✅ Extracted %d items via regex.", len(extracted))
            return extracted

    # 5) Helpers for ranking step -------------------------------------- #
    def get_opposites(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Return opposite/contradictory concepts for each keyword."""
        prompt = f"""
                    You are a materials science assistant.

                    For each of the following scientific keywords, return a list of their
                    opposite or contradictory concepts in the context of electrocatalysis,
                    water splitting, or electrochemical environments.

                    Keywords: {keywords}

                    Respond only in JSON format like:
                    {{ "acidic": ["alkaline", "basic"], ... }}
                """
        response = self.ask(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            LOG.warning("⚠️ Could not parse opposites JSON. Raw:\n%s", response)
            return {}

    def get_synonyms(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Return synonyms or semantically similar terms for each keyword."""
        prompt = f"""
                    You are a materials science assistant.

                    For each of the following scientific keywords, return a list of their
                    synonyms or semantically similar expressions used in electrocatalysis,
                    water splitting, or electrochemistry papers.

                    Keywords: {keywords}

                    Respond only in JSON format like:
                    {{ "acidic": ["low pH", "acid media"], ... }}
                """
        response = self.ask(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            LOG.warning("⚠️ Could not parse synonym JSON. Raw:\n%s", response)
            return {}
