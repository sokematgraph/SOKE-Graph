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
import torch
from typing import Dict, List

LOG = logging.getLogger(__name__)


class AIAgent(ABC):

    """Abstract base class for any large‑language‑model agent.

    Sub‑classes (e.g. `OpenAIAgent`, `GeminiAgent`, `LlamaAgent`) must
    implement :py:meth:`ask`, which sends a prompt to the underlying model
    and returns the raw text response.
    """


    def __init__(self, field_of_interest=None):
        """
        Set device for GPU if available.
        - On Mac: uses MPS (Metal)
        - Otherwise: CUDA if available
        - Fallback: CPU
        """
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            LOG.info("Using MPS GPU for AI agent.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            LOG.info("Using CUDA GPU for AI agent.")
        else:
            self.device = torch.device("cpu")
            LOG.info("Using CPU for AI agent.")

        self.field_of_interest = field_of_interest

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
        """Create a strict extraction prompt that demands pure JSON (no code fences)."""
        return (
            f"""
    You are a structured information extraction AI specialized in materials science.

    TASK: Extract information for the ontology layer: "{layer_name}".

    ONTOLOGY LAYER (categories → sample keywords):
    {json.dumps(ontology_layer, ensure_ascii=False, indent=2)}

    TEXT TO ANALYZE:
    {abstract_text}

    OUTPUT FORMAT (IMPORTANT):
    - OUTPUT MUST BE PURE JSON (a JSON array). DO NOT include code fences, markdown, or any text outside the JSON.
    - EACH ITEM must follow this exact schema:
      {{
        "layer": "{layer_name}",
        "matched_category": "<one of the categories exactly as shown above>",
        "keyword": "<extracted keyword or phrase>",
        "meta_data": "<short supporting snippet from the text>"
      }}
    - Keep 1–25 items. Use lowercase for keywords where natural.
    - If nothing matches, return [] (empty JSON array).

    EXAMPLES (DO NOT ECHO; JUST FOLLOW THE FORMAT):
    Example:
    [
      {{
        "layer": "{layer_name}",
        "matched_category": "<CategoryName>",
        "keyword": "iridium oxide",
        "meta_data": "… overpotential of 240 mV at 10 mA cm^-2 in acidic media …"
      }}
    ]

    NOW RETURN ONLY THE JSON ARRAY as a string don't include any characters such as ``` or json or any sort of code formatting.
    """.strip()
        )

    def _strip_fences(self, s: str) -> str:
        """Remove accidental markdown fences/backticks/whitespace around JSON."""
        if not isinstance(s, str):
            return s
        s = s.strip()
        # Remove ```json ... ``` or ``` ... ```
        if s.startswith("```"):
            s = s.lstrip("`")
            # common 'json' language tag
            if s.lower().startswith("json"):
                s = s[4:].lstrip()
            s = s.rstrip("`").strip()
        # Also trim stray leading/trailing backticks
        return s.strip("`").strip()

    def _safe_json_loads(self, raw: str):
        """Try to parse JSON after stripping common wrappers; return None on failure."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # try again after stripping common wrappers
            cleaned = self._strip_fences(raw)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def get_opposites(self, keywords: list[str]) -> dict[str, list[str]]:
        """Return opposite/contradictory concepts for each keyword (pure JSON only)."""
        keywords_json = json.dumps(keywords, ensure_ascii=False)

        prompt = f"""
    You are a {self.field_of_interest} assistant.
    TASK: For each keyword in the list, return a JSON object mapping the keyword (as given) to a list of opposite or contradictory concepts in the context of electrocatalysis, water splitting, and electrochemical environments.

    RESPONSE RULES (IMPORTANT):
    - OUTPUT MUST BE PURE JSON (a single JSON object). DO NOT include code fences, backticks, markdown, comments, or any text outside the JSON.
    - Keys MUST be exactly the input keywords (case preserved). Do not add or remove keys.
    - Values MUST be lists of lowercase strings (2–6 items). Deduplicate terms.
    - Prefer domain-specific opposites:
      • device: pemwe ↔ pemfc/aemwe
      • environment: acidic ↔ alkaline/basic
      • reaction: oer ↔ her/orr
      • catalyst class: earth-abundant/non-noble ↔ noble/precious/pgm
    - If no good opposite exists, return an empty list for that key.

    FEW-SHOT EXAMPLES (for guidance only; DO NOT echo these examples):
    Example A (device):
    Input: ["pemwe","pem water electrolyzer"]
    Output: {{"pemwe": ["pemfc","fuel cell","aemwe","alkaline electrolyzer"], "pem water electrolyzer": ["pemfc","fuel cell","aemwe","alkaline water electrolyzer"]}}

    Example B (environment):
    Input: ["acidic"]
    Output: {{"acidic": ["alkaline","basic","koh","naoh"]}}

    Example C (reactions):
    Input: ["oer","oxygen evolution"]
    Output: {{"oer": ["her","hydrogen evolution","orr","oxygen reduction"], "oxygen evolution": ["hydrogen evolution","her","oxygen reduction","orr"]}}

    Example D (catalyst class):
    Input: ["earth-abundant","non-noble"]
    Output: {{"earth-abundant": ["noble","precious","pgm","platinum","pt","iridium","ir","ruthenium","ru"], "non-noble": ["noble","precious","pgm","pt","ir","ru"]}}

    YOUR INPUT:
    {keywords_json}

    Return ONLY the JSON object for these keywords, following the rules.
    """.strip()

        raw = self.ask(prompt)
        obj = self._safe_json_loads(raw)

        if obj is None or not isinstance(obj, dict):
            LOG.warning("⚠️ Could not parse opposites JSON. Raw:\n%s", raw)
            return {}

        # Post-process: normalize values to lowercase unique strings; ensure lists
        cleaned = {}
        for k in keywords:
            vals = obj.get(k, [])
            if not isinstance(vals, list):
                vals = []
            cleaned[k] = sorted({str(v).strip().lower() for v in vals if str(v).strip()})
        LOG.info("✅ Opposites JSON parsed for %d keys.", len(cleaned))
        return cleaned

    def get_synonyms(self, keywords: list[str]) -> dict[str, list[str]]:
        """Return synonyms / semantically similar terms for each keyword (pure JSON only)."""
        keywords_json = json.dumps(keywords, ensure_ascii=False)

        prompt = f"""
    You are a {self.field_of_interest} assistant.
    TASK: For each keyword in the list, return a JSON object mapping the keyword (as given) to a list of synonyms or semantically similar expressions used in electrocatalysis / water splitting / electrochemistry papers.

    RESPONSE RULES (IMPORTANT):
    - OUTPUT MUST BE PURE JSON (a single JSON object). DO NOT include code fences, backticks, markdown, comments, or any text outside the JSON.
    - Keys MUST be exactly the input keywords (case preserved). Do not add or remove keys.
    - Values MUST be lists of lowercase strings (2–10 items). Deduplicate.
    - Include common abbreviations and formula forms (e.g., h2so4, iro2, pt).
    - Do NOT include antonyms/opposites.
    - Prefer tokens that are directly usable for exact/regex matching.

    FEW-SHOT EXAMPLES (for guidance only; DO NOT echo these examples):
    Example A:
    Input: ["acidic"]
    Output: {{"acidic": ["acid","acid media","low ph","protonic","h2so4","sulfuric acid"]}}

    Example B:
    Input: ["pemwe","pem water electrolyzer"]
    Output: {{"pemwe": ["polymer electrolyte membrane water electrolyzer","proton exchange membrane water electrolyzer","pem water electrolyzer"], "pem water electrolyzer": ["pemwe","polymer electrolyte membrane water electrolyzer","proton exchange membrane water electrolyzer"]}}

    Example C:
    Input: ["oer","oxygen evolution"]
    Output: {{"oer": ["oxygen evolution reaction","water oxidation","anodic water oxidation"], "oxygen evolution": ["oer","water oxidation","anodic oxygen evolution"]}}

    Example D:
    Input: ["earth-abundant","pgm-free"]
    Output: {{"earth-abundant": ["base-metal","non-noble","pgm-free","precious-metal-free","low-cost"], "pgm-free": ["non-noble","precious-metal-free","earth-abundant"]}}

    YOUR INPUT:
    {keywords_json}

    Return ONLY the JSON object for these keywords, following the rules.
    """.strip()

        raw = self.ask(prompt)
        obj = self._safe_json_loads(raw)

        if obj is None or not isinstance(obj, dict):
            LOG.warning("⚠️ Could not parse synonym JSON. Raw:\n%s", raw)
            return {}

        # Post-process: normalize values to lowercase unique strings; ensure lists
        cleaned = {}
        for k in keywords:
            vals = obj.get(k, [])
            if not isinstance(vals, list):
                vals = []
            cleaned[k] = sorted({str(v).strip().lower() for v in vals if str(v).strip()})
        LOG.info("✅ Synonym JSON parsed for %d keys.", len(cleaned))
        return cleaned

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
                print(f"  - Layer: {layer} abstract : {abstract} categories : {categories}")
                results = self._call_model(layer, abstract, categories)
                print(f"results : {results} for paper_id : {paper_id}")
                # Check if results is not empty
                if len(results) > 0:
                    for res in results:
                        cat = res["matched_category"]
                        if cat not in ontology[layer]:
                            LOG.warning(f"⚠️ Skipping unknown category '{cat}' under layer '{layer}'",
                                        cat, layer)
                            continue
                print(f"results : {results} for paper_id : {paper_id}")
                # Check if results is not empty
                if len(results) > 0:
                    for res in results:
                        cat = res["matched_category"]
                        if cat not in ontology[layer]:
                            LOG.warning(f"⚠️ Skipping unknown category '{cat}' under layer '{layer}'",
                                        cat, layer)
                            continue

                        # Deduplicate keywords
                        keywords = list(set(ontology[layer][cat] + [res["keyword"]]))
                        results_dict[layer][cat].append({
                            "keywords": keywords,
                            "meta_data": res["meta_data"],
                            "paper_id": paper_id
                        })

                    LOG.info(f"✅ Total items extracted for paper {paper_id}, layer '{layer}': {len(results)}")
                else:
                    LOG.warning(f"⚠️ No items extracted for paper {paper_id}, layer '{layer}'")
        return results_dict

    # 4) Private helpers ------------------------------------------------ #
    def _call_model(self, layer: str,
                    abstract_text: str,
                    ontology_layer: dict) -> List[dict]:
        """Utility that builds the prompt, queries the model and parses."""
        import numpy as np
        if(abstract_text is  np.nan):
            abstract_text = ""  
            
        prompt = self._build_prompt(layer, abstract_text, ontology_layer)
        print(f"prompt : {prompt}")
        print("xxxx")
        model_output = self.ask(prompt)
        print(f"model_output : {model_output}")
        items = [] if not model_output else self._parse_model_output(model_output)
        return items

        # 4) Private helpers ------------------------------------------------ #
    def _parse_model_output(self, model_output: str) -> List[dict]:
        """Parse model output—JSON first (with fence stripping), regex fallback.

        Returns:
            A list[dict] with keys: layer, matched_category, keyword, meta_data.
        """
        # ---- JSON-first path (reuses the same tolerant loader as synonyms/opposites)
        obj = self._safe_json_loads(model_output)
        if obj is not None:
            # Case 1: model returned the expected JSON array directly
            if isinstance(obj, list):
                return self._normalize_items_list(obj)

            # Case 2: some models wrap results inside an object; try to find an array value
            if isinstance(obj, dict):
                # Prefer a value that looks like the extraction list
                for v in obj.values():
                    if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                        return self._normalize_items_list(v)

        # ---- Try to extract a JSON array substring and parse again
        cleaned = self._strip_fences(model_output)
        m = re.search(r"\[\s*{.*}\s*\]", cleaned, flags=re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return self._normalize_items_list(arr)
            except json.JSONDecodeError:
                pass

        # ---- Final fallback: regex
        LOG.warning("⚠️ JSON parsing failed. Attempting fallback regex...")
        pattern = r'"layer":\s*"([^"]+)",\s*"matched_category":\s*"([^"]+)",\s*"keyword":\s*"([^"]+)",\s*"meta_data":\s*"([^"]+)"'
        matches = re.findall(pattern, model_output, re.DOTALL)
        extracted = [
            dict(layer=m[0].strip(),
                 matched_category=m[1].strip(),
                 keyword=m[2].strip(),
                 meta_data=m[3].strip())
            for m in matches
        ]
        LOG.info("✅ Extracted %d items via regex.", len(extracted))
        return extracted

    # ---- Small helper to validate/normalize the parsed list-of-dicts
    def _normalize_items_list(self, items: List[dict]) -> List[dict]:
        """Ensure each item has expected keys; coerce to strings and strip."""
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            out.append({
                "layer": str(it.get("layer", "")).strip(),
                "matched_category": str(it.get("matched_category", "")).strip(),
                "keyword": str(it.get("keyword", "")).strip(),
                "meta_data": str(it.get("meta_data", "")).strip(),
            })
        LOG.info("✅ JSON parsed successfully (%d items).", len(out))
        return out
