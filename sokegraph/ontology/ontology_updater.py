"""
ontology_updater.py

Provides OntologyUpdater — a helper that enriches an ontology JSON file
using AI‑extracted keywords from a list of papers, then persists the updated
ontology to disk.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Dict
import pandas as pd
from pathlib import Path

from sokegraph.agents.ai_agent import AIAgent
from sokegraph.utils.functions import parse_all_metadata, load_papers


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
        papers_path: str,
        ai_tool: AIAgent,
        output_dir: str,
    ):
        self.ontology_path = ontology_path
        self.papers = pd.DataFrame(load_papers(papers_path))
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
        # print(f"text_data :{text_data}")
        # Ask the AI agent to extract keywords & merge into ontology
        self.ontology = self.ai_tool.extract_keywords(self.ontology, text_data)

        # Parse numeric metadata strings (overpotential, Tafel slope, etc.)
        self.ontology = parse_all_metadata(self.ontology)

        # Persist to disk
        self.output_path = os.path.join(self.output_dir, "updated_ontology.json")
        self._save_ontology(self.output_path)

        return self.output_path

    def update_base_ontology(self, user_query: str) -> None:
        """ Update the base ontology, overwrite the file """
        return None

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
        print(f"papers :{self.papers} ")
        for _, row in self.papers.iterrows():  # row is a pd.Series
            # print(f"row: {row.to_dict()}")

            safe_id = str(row.get("paper_id", ""))

            text_data[safe_id] = row.get("abstract", "")

        return text_data

    # ─────────────────────────────────────────────────────────────────────
    # NEW: local base-ontology enrichment (in-place, path unchanged)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def enrich_base_with_keywords(
            base_ontology_path: str,
            keywords: List[str],
            ai_tool: Any,
            prompt_overrides: str | None = None,
    ) -> None:
        """
        Read base ontology JSON from disk, call AI to map user keywords into the
        ontology shape, merge results, and WRITE BACK TO THE SAME FILE PATH.

        - Preserves format (flat lists vs. dict-of-subsections).
        - Idempotent (case-insensitive de-dupe).
        - Robust: if AI returns nothing/invalid JSON, append to Keyword->UserProvided.
        - Writes only if new items were actually added (avoids 'reformat-only' rewrites).
        """
        path = Path(base_ontology_path)
        if not path.exists():
            raise FileNotFoundError(f"Base ontology not found: {path}")

        # Load current ontology
        root = json.loads(path.read_text(encoding="utf-8"))
        has_data_wrapper = isinstance(root, dict) and "data" in root and isinstance(root["data"], dict)
        tree: Dict[str, Any] = root["data"] if has_data_wrapper else root

        # ---- Helpers (merge & slots) ----
        def _merge_list(dst: List[str], src: List[str]) -> int:
            added = 0
            seen = {str(x).strip().lower() for x in dst}
            for item in src or []:
                s = str(item).strip()
                if not s:
                    continue
                low = s.lower()
                if low not in seen:
                    dst.append(s)
                    seen.add(low)
                    added += 1
            return added

        def _ensure_keyword_bucket() -> List[str]:
            """Return a writable list for Keyword safe-bucket."""
            if "Keyword" not in tree:
                tree["Keyword"] = {"UserProvided": []}
            if isinstance(tree["Keyword"], dict):
                tree["Keyword"].setdefault("UserProvided", [])
                return tree["Keyword"]["UserProvided"]
            if isinstance(tree["Keyword"], list):
                return tree["Keyword"]
            # unexpected shape → reset to safe dict
            tree["Keyword"] = {"UserProvided": []}
            return tree["Keyword"]["UserProvided"]

        def _ensure_list_slot(top: str, sub: str | None = None) -> List[str]:
            """Return a writable list at tree[top] (flat) or tree[top][sub] (hier)."""
            if sub is None:
                # flat section
                if top not in tree or not isinstance(tree[top], (list, dict)):
                    tree[top] = []
                if isinstance(tree[top], list):
                    return tree[top]
                # if dict, use UserProvided subsection
                tree[top].setdefault("UserProvided", [])
                return tree[top]["UserProvided"]
            # hierarchical section
            tree.setdefault(top, {})
            if not isinstance(tree[top], dict):
                tree[top] = {"UserProvided": []}
            tree[top].setdefault(sub, [])
            return tree[top][sub]

        def _flatten_payload(payload) -> List[str]:
            """Flatten dict-of-lists or list payloads into a simple list of strings."""
            out: List[str] = []
            if isinstance(payload, list):
                out.extend([x for x in payload if isinstance(x, (str, int, float))])
            elif isinstance(payload, dict):
                for v in payload.values():
                    if isinstance(v, list):
                        out.extend([x for x in v if isinstance(x, (str, int, float))])
            return [str(x).strip() for x in out if str(x).strip()]

        # ---- Pre-router (cheap heuristics for common phrases) ----
        total_added = 0
        for kw in [str(k).strip() for k in (keywords or []) if str(k).strip()]:
            kw_l = kw.lower()

            # Environment
            if "acidic" in kw_l or ("ph" in kw_l and "<" in kw_l):
                total_added += _merge_list(_ensure_list_slot("Environment", "Acidic"), ["acidic"])
            if "alkaline" in kw_l or "basic" in kw_l or ("ph" in kw_l and ">" in kw_l):
                total_added += _merge_list(_ensure_list_slot("Environment", "Alkaline"), ["alkaline"])
            if "neutral" in kw_l or "ph 7" in kw_l:
                total_added += _merge_list(_ensure_list_slot("Environment", "Neutral"), ["neutral"])

            # Reaction
            if (" her " in f" {kw_l} ") or kw_l.endswith(" her") or kw_l.startswith(
                    "her ") or "hydrogen evolution" in kw_l:
                total_added += _merge_list(_ensure_list_slot("Reaction", "Hydrogen Evolution Reaction (HER)"), ["HER"])
            if (" oer " in f" {kw_l} ") or kw_l.endswith(" oer") or kw_l.startswith(
                    "oer ") or "oxygen evolution" in kw_l:
                total_added += _merge_list(_ensure_list_slot("Reaction", "Oxygen Evolution Reaction (OER)"), ["OER"])

            # Process
            if "water splitting" in kw_l or "electrolysis" in kw_l:
                total_added += _merge_list(_ensure_list_slot("Process", "Water Electrolysis"), ["water splitting"])
            if "photoelectrochemical" in kw_l or " pec " in f" {kw_l} ":
                total_added += _merge_list(_ensure_list_slot("Process", "Photoelectrochemical Water Splitting"),
                                           ["photoelectrochemical"])
            if "photocatalysis" in kw_l or "photocatalyst" in kw_l:
                total_added += _merge_list(_ensure_list_slot("Process", "Photocatalysis"), ["photocatalysis"])
            if "fuel cell" in kw_l or " pem " in f" {kw_l} ":
                total_added += _merge_list(_ensure_list_slot("Process", "Fuel Cells"), ["fuel cell"])

        # ---- Shape hints for the LLM (compact; no full JSON) ----
        base_keys = list(tree.keys())
        shape_hints = {}
        for k, v in tree.items():
            if isinstance(v, list):
                shape_hints[k] = {"type": "list"}
            elif isinstance(v, dict):
                shape_hints[k] = {"type": "dict", "subsections": list(v.keys())[:50]}
            else:
                shape_hints[k] = {"type": type(v).__name__}

        # Tiny examples to improve routing (keep small)
        examples: Dict[str, Any] = {}
        for top, slot in tree.items():
            if isinstance(slot, dict):
                ex = {}
                for sub_name, sub_list in list(slot.items())[:6]:
                    ex[sub_name] = list(sub_list)[:2] if isinstance(sub_list, list) else []
                examples[top] = ex
            elif isinstance(slot, list):
                examples[top] = list(slot)[:6]

        # ---- Prompt (require 'additions' at top-level; matches our parser) ----
    #     prompt = prompt_overrides or f"""
    # You are an ontology assistant. You will receive (a) SHAPE HINTS for the base ontology,
    # (b) a few tiny EXAMPLES, and (c) user-provided keywords. Produce a SMALL JSON *patch*.
    #
    # RETURN **EXACTLY** THIS TOP-LEVEL SHAPE:
    # {{
    #   "additions": {{
    #     // For flat sections (LIST):
    #     //   "<TopLevel>": ["term1", "term2"]
    #     //
    #     // For hierarchical sections (DICT of subsections):
    #     //   "<TopLevel>": {{
    #     //     "<Subsection>": ["term1", "term2"]
    #     //   }}
    #   }}
    # }}
    #
    # ROUTING RULES:
    # - Allowed top-level sections ONLY: {base_keys}
    # - Section shapes (list vs dict, sample subsection names):
    #   {json.dumps(shape_hints, ensure_ascii=False)}
    # - Use the EXAMPLES as guidance for terminology and placement (do not rewrite them):
    #   {json.dumps(examples, ensure_ascii=False)}
    # - Split composite phrases into atomic concepts and route EACH concept.
    # - If unsure where to put a term, use "Keyword" -> "UserProvided".
    # - ONLY ADD strings. Do not rename/remove anything.
    # - Avoid duplicates. Keep strings short/plain.
    #
    # USER_KEYWORDS: {keywords}
    # """.strip()

        prompt = prompt_overrides or f"""
        You are an ontology assistant. You will receive:
        1) SHAPE HINTS for the base ontology,
        2) EXAMPLES of existing terms, and
        3) USER KEYWORDS (queries).

        TASK:
        - Split each keyword/query into atomic concepts.
        - Map each concept into the MOST RELEVANT bucket of the ontology (top-level + subsection).
        - Use ONLY the existing top-level sections: {base_keys}
        - Section shapes: {json.dumps(shape_hints, ensure_ascii=False)}
        - Use these EXAMPLES for guidance (do not modify them):
          {json.dumps(examples, ensure_ascii=False)}

        OUTPUT (return STRICT JSON in this exact format):
        {{
          "additions": {{
            "<TopLevel>": {{
              "<Subsection>": ["new_term1", "new_term2"]
            }},
            "<AnotherTopLevel>": ["new_term3", "new_term4"]
          }}
        }}

        RULES:
        - DO NOT return the full ontology, only the "additions".
        - DO NOT put the entire query as one string.
        - Route terms to the right bucket: e.g.
          "acidic earth abundant electrochemical catalysts for water splitting"
           → {{
                "Environment": {{"Acidic": ["acidic"]}},
                "Elemental Composition": {{"Earth-Abundant Elements": ["earth abundant"]}},
                "Process": {{"Water Electrolysis": ["water splitting"]}},
                "Process": {{"Electrocatalysis": ["electrochemical catalysts"]}}
              }}
        - If a concept clearly fits multiple buckets, include it in each.
        - If you cannot decide, use "Keyword" -> "UserProvided".
        """.strip()
        # ---- Call the AI (flexible interface) ----
        def _call_ai(agent: Any, text: str) -> str:
            for name in ("ask", "generate", "complete", "invoke", "run", "__call__"):
                if hasattr(agent, name):
                    fn = getattr(agent, name)
                    try:
                        return fn(text)
                    except TypeError:
                        continue
            return agent(text)  # last resort

        additions: Dict[str, Dict[str, List[str]]] | List[str] | None = None
        try:
            raw = _call_ai(ai_tool, prompt)
            parsed = json.loads(raw)
            additions = parsed.get("additions") if isinstance(parsed, dict) else None
        except Exception:
            additions = None  # fall back below

        # ---- Merge AI additions or fall back ----
        if not additions or not isinstance(additions, dict):
            bucket = _ensure_keyword_bucket()
            total_added += _merge_list(bucket, [str(k).strip() for k in (keywords or []) if str(k).strip()])
        else:
            allowed = set(base_keys)
            for top, payload in additions.items():
                if top not in allowed:
                    bucket = _ensure_keyword_bucket()
                    total_added += _merge_list(bucket, _flatten_payload(payload))
                    continue

                slot = tree.get(top)

                # PATCH is a LIST (flat append)
                if isinstance(payload, list):
                    if isinstance(slot, list):
                        total_added += _merge_list(slot, payload)
                    elif isinstance(slot, dict):
                        slot.setdefault("UserProvided", [])
                        total_added += _merge_list(slot["UserProvided"], payload)
                    else:
                        tree[top] = []
                        total_added += _merge_list(tree[top], payload)

                # PATCH is a DICT (subsections)
                elif isinstance(payload, dict):
                    if isinstance(slot, dict):
                        for subsection, terms in payload.items():
                            if not isinstance(terms, list):
                                continue
                            slot.setdefault(subsection, [])
                            total_added += _merge_list(slot[subsection], terms)
                    elif isinstance(slot, list):
                        total_added += _merge_list(slot, _flatten_payload(payload))
                    else:
                        tree[top] = {}
                        for subsection, terms in payload.items():
                            if not isinstance(terms, list):
                                continue
                            tree[top].setdefault(subsection, [])
                            total_added += _merge_list(tree[top][subsection], terms)
                # else: ignore unknown payload types

        # ---- Write back only if something changed ----
        if total_added > 0:
            payload = {"data": tree} if has_data_wrapper else tree
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

