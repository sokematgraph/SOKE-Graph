"""
paper_ranker.py

Implements :class:`PaperRanker`, which scores a list of research papers
for relevance to a user keyword query using:

1. An ontology of layers → categories → keywords
2. An :class:`~sokegraph.ai_agent.AIAgent` to classify the query and expand
   keywords (synonyms / opposites)
3. Keyword-mention counts in titles & abstracts (Static)
4. IDF-weighted dynamic scoring with adaptive threshold (Dynamic)
5. Neural Hybrid Relevance Model (HRM): learned scorer over ontology + metadata tokens

CSV reports are exported to `output_dir`.

Notes:
- If PyTorch is unavailable or no HRM weights / labels are provided, the code
  falls back to a simple HRM that works on current ontology's category and layer to stay compatible.
"""


import re
import math
import csv
import random
import pandas as pd
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Any, Tuple, Set, Optional

from sokegraph.util.logger import LOG
from sokegraph.utils.functions import load_keyword, safe_title
from sokegraph.agents.ai_agent import AIAgent

import os
import json
import pandas as pd
import pyarrow.parquet as pq
from rdflib import Graph, Literal, RDF, URIRef, Namespace
import networkx as nx
from sokegraph.utils.functions import load_papers


# ----------------------
# Optional PyTorch import
# ----------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ----------------------
# Optional BM25 import
# ----------------------
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False

class PaperRanker:
    def __init__(
        self,
        ai_tool: AIAgent,
        papers_path,
        ontology_path,
        keyword_path,
        output_dir: str,
        hrm_alpha: float = 1.0,
        hrm_beta: float = 1.0,
        # -------- HRM CONFIG (adjust paths as needed) --------
        hrm_use_neural: bool = True,
        hrm_vocab_path: Optional[str] = None,       # optional JSON {token: idx}
        hrm_weights_path: Optional[str] = None,     # optional .pt to load a trained HRM
        hrm_labels_path: Optional[str] = None,      # optional CSV with supervision for training
        hrm_train_epochs: int = 40,
        hrm_batch_size: int = 64,
        hrm_lr: float = 1e-3,
        hrm_max_seq_len: int = 64,
        hrm_dim: int = 128,
        hrm_halting_max_steps: int = 3,
        hrm_seed: int = 1337,
        bypass_filtering: bool = False,
        # -------- FALLBACK CONFIG --------
        hrm_use_simple_fallback: bool = False,      # Use simple fallback from paper_ranker.py.txt
        polarity_lambda: float = 0.5,               # For simple fallback penalty calculation
        # -------- LAYER PRIORITY CONFIG --------
        layer_priority_weights: Optional[Dict[str, float]] = None,  # Layer priority multipliers
    ):
        """
        Initialize the PaperRanker.

        Parameters
        ----------
        ai_tool : AIAgent
            AI agent for classification and synonym/opposite expansion.
        papers : list[dict]
            List of paper metadata (must contain 'title', 'abstract', 'paper_id').
        ontology_path : str
            Path to the ontology JSON file.
        keyword_path : str
            Path to the user keyword query text file.
        output_dir : str
            Directory where output CSVs will be saved.
        hrm_alpha, hrm_beta : float
            Weights for fallback simple HRM if neural HRM is unavailable.
        hrm_use_neural : bool
            If True and PyTorch available, use Neural HRM (else fallback).
        hrm_* : various
            Training/inference config for Neural HRM.
        hrm_use_simple_fallback : bool
            If True, use the simpler HRM fallback from paper_ranker.py.txt instead of complex one.
        polarity_lambda : float
            Lambda parameter for simple fallback penalty calculation.
        layer_priority_weights : Optional[Dict[str, float]]
            Dictionary mapping ontology layer names to priority multipliers.
            Higher values boost papers matching that layer.
            Default: {"Device": 3.0, "Environment": 2.5, "Elemental Composition": 2.0}
            Unlisted layers default to 1.0 (neutral).
        """
        self._static_text_terms = None
        self._ontology_matched_terms = None
        self.ai_tool = ai_tool
        self.papers = load_papers(papers_path)
        self.ontology_path = ontology_path
        self.ontology = None  # set by _load_ontology
        self._load_ontology()
        self.output_dir = output_dir
        self.keyword_query = load_keyword(keyword_path)

        self._debug_matches = {"static": {}, "dynamic": {}, "hrm": {}}
        self._occ_detail = {} # this counts how many times each term occured
        self.bypass_filtering = bypass_filtering

        # HRM configs
        self.hrm_alpha = hrm_alpha
        self.hrm_beta = hrm_beta
        self.hrm_use_neural = bool(hrm_use_neural and TORCH_AVAILABLE)
        self.hrm_vocab_path = hrm_vocab_path
        self.hrm_weights_path = hrm_weights_path
        self.hrm_labels_path = hrm_labels_path
        self.hrm_train_epochs = hrm_train_epochs
        self.hrm_batch_size = hrm_batch_size
        self.hrm_lr = hrm_lr
        self.hrm_max_seq_len = hrm_max_seq_len
        self.hrm_dim = hrm_dim
        self.hrm_halting_max_steps = hrm_halting_max_steps
        self.hrm_seed = hrm_seed

        # Fallback config
        self.hrm_use_simple_fallback = hrm_use_simple_fallback
        self.polarity_lambda = polarity_lambda

        # Layer priority config
        self.layer_priority_weights = layer_priority_weights or {
            "Device": 3.0,
            "Environment": 2.5,
            "Elemental Composition": 2.0
        }

        # HRM runtime
        self._hrm_vocab = None
        self._hrm_model = None
        self._hrm_device = "cpu"

        # HRM (paper-style) extras
        self.hrm_N = getattr(self, "hrm_N", 3)          # high-level cycles
        self.hrm_T = getattr(self, "hrm_T", 4)          # low-level steps per cycle
        self.hrm_use_act = True                         # enable ACT/Q-head
        self.hrm_Mmax = 4                               # max segments for deep supervision/ACT
        self.hrm_eps_minseg = 0.25                      # epsilon for random longer min-segment

        self.static_lambda = 0.4
        self.static_alpha = 100.0

        # --- By pass union ---
        self.bypass_union_for_hrm = True
        self.bypass_union_for_table = True
        self.bypass_union_for_pairs = True

        if self.hrm_use_neural:
            self._prepare_neural_hrm_runtime()

    # --------------------
    # ONTOLOGY & RANK FLOW
    # --------------------

    def _build_category_terms(
            self,
            categories,  # list of (layer, cat, kw) selected by classify_query_with_fallback
            query_keywords,  # list[str]
            synonyms: Dict[str, List[str]]
    ) -> Dict[Tuple[str, str], Set[str]]:
        """
        Map each (layer, category) to a set of expanded terms (keyword + synonyms) that
        belong to that category, based on the AI classification.
        """
        cat_to_terms: Dict[Tuple[str, str], Set[str]] = {}
        for (layer, cat, kw) in categories:
            terms = {kw.lower()}
            for s in (synonyms.get(kw.lower(), []) or []):
                if s: terms.add(s.lower())
            key = (layer, cat)
            cat_to_terms.setdefault(key, set()).update(terms)
        return cat_to_terms

    def _count_relevant_opposing(
            self,
            title_text: str,
            abstract_text: str,
            query_keywords: List[str],
            expanded_keywords: Dict[str, List[str]],
            opposites: Dict[str, List[str]],
    ) -> tuple[int, int]:
        """
        Total relevant and opposing hits across all query terms (title + abstract).
        Uses expanded keywords per query term and their 'opposites'.
        """
        import re
        text = f"{(title_text or '').lower()} {(abstract_text or '').lower()}"

        rel_concepts_hit = set()
        opp_concepts_hit = set()

        for qk in query_keywords:
            # relevant: binary presence per concept
            rel_terms = {qk} | set((expanded_keywords.get(qk, []) or []))
            if any(term and re.search(rf"\b{re.escape(term)}\b", text) for term in rel_terms):
                rel_concepts_hit.add(qk)

            # opposite: binary presence per concept
            opp_terms = set((opposites.get(qk, []) or []))
            if any(term and re.search(rf"\b{re.escape(term)}\b", text) for term in opp_terms):
                opp_concepts_hit.add(qk)

        return len(rel_concepts_hit), len(opp_concepts_hit)

    def _load_ontology(self):
        """Load ontology JSON from self.ontology_path into self.ontology."""
        try:
            with open(self.ontology_path, "r", encoding="utf-8") as f:
                self.ontology = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load ontology from '{self.ontology_path}': {e}")

    def rank_papers(self) -> Dict[str, Any]:
        """
        Runs three scoring tracks and returns CSV paths as a flat dict:
          - static_* : static scoring (keyword mention counts)
          - dynamic_* : IDF-weighted + adaptive threshold
          - hrm_* : Neural HRM (or simple fallback HRM if neural not available)
        """

        # Extract mappings
        title_map = self._get_title_map()
        print(f"title_map : {title_map}")
        abstract_map = self._get_abstract_map()
        print(f"abstract_map : {abstract_map}")

        # Build category/paper keyword maps
        category_to_papers = defaultdict(lambda: defaultdict(int))
        paper_keyword_map = defaultdict(lambda: defaultdict(set))
        kw_lookup = {}

        for layer, categories in self.ontology.items():
            for category, items in categories.items():
                for item in items:
                    paper_id = item["paper_id"]
                    print(f"item {paper_id}")
                    for kw in item["keywords"]:
                        kw_lower = kw.lower()  # Normalize keyword to lowercase
                        kw_lookup[kw_lower] = (layer, category)

                        # Track which keywords are linked to this paper in this category
                        paper_keyword_map[(layer, category)][paper_id].add(kw_lower)

                        # Increment count of how many times this paper maps to this category
                        category_to_papers[(layer, category)][paper_id] += 1

        # Find candidate papers and scores (both static & dynamic)
        static_ranked, dynamic_ranked, low_sorted, pair_file_path, pair_counts = self.find_common_papers(
            category_to_papers,
            paper_keyword_map,
            title_map,
            kw_lookup,
            abstract_map,
        )

        # Write ranked CSVs per method
        static_outputs_csv,  static_outputs_all = self.rank_shared_papers_by_pairs_and_mentions(
            static_ranked, low_sorted, pair_file_path, self.output_dir, label="static"
        )
        dynamic_outputs_csv,  dynamic_outputs_all = self.rank_shared_papers_by_pairs_and_mentions(
            dynamic_ranked, low_sorted, pair_file_path, self.output_dir, label="dynamic"
        )

        # HRM: Neural (preferred) or Hierarchical-Reasoning fallback
        if self.hrm_use_neural and self._hrm_model is not None and len(pair_counts) > 0:
            hrm_ranked = self._score_with_neural_hrm(
                static_ranked=static_ranked,
                dynamic_ranked=dynamic_ranked,
                pair_counts=pair_counts,
                title_map=title_map,
                abstract_map=abstract_map,
                paper_keyword_map=paper_keyword_map,
            )
        else:
            # Use union of survivors as the candidate pool for HRS
            candidate_ids = sorted({pid for pid, _ in static_ranked} | {pid for pid, _ in dynamic_ranked})
            LOG.info(f"the Union of Candidate IDs: {candidate_ids}")

            if self.hrm_use_simple_fallback:
                LOG.info("HRM fallback: using Simple HRM fallback (from paper_ranker.py.txt)")
                hrm_ranked = self._score_with_simple_hrm_fallback(
                    candidate_ids=candidate_ids,
                    title_map=title_map,
                    abstract_map=abstract_map,
                    paper_keyword_map=paper_keyword_map,
                    kw_lookup=kw_lookup,
                )
            else:
                LOG.info("HRM fallback: using Complex Hierarchical Reasoning Score (HRS)")
                hrm_ranked = self._score_with_HRM(
                    candidate_ids=candidate_ids,
                    title_map=title_map,
                    abstract_map=abstract_map,
                    paper_keyword_map=paper_keyword_map,
                    kw_lookup=kw_lookup,
                )

        hrm_outputs_csv,  hrm_outputs_all = self.rank_shared_papers_by_pairs_and_mentions(
            hrm_ranked, low_sorted, pair_file_path, self.output_dir, label="hrm"
        )

        # (Optional) write comparison table for records (does not change return shape)
        try:
            self._write_comparison_table(
                candidate_ids=sorted({pid for pid, _ in static_ranked} | {pid for pid, _ in dynamic_ranked}),
                static_ranked=static_ranked,
                dynamic_ranked=dynamic_ranked,
                hrs_ranked=hrm_ranked,
                output_path=os.path.join(self.output_dir, "comparison_table.csv"),
            )
            LOG.info("📄 Saved comparison table to 'comparison_table.csv'")
        except Exception as e:
            LOG.warning(f"Could not write comparison table: {e}")

        # Backward-compatible: return a flat dict of CSV paths
        results_csv = static_outputs_csv.copy()
        results_csv.update(dynamic_outputs_csv)
        results_all = static_outputs_all.copy()
        results_all.update(dynamic_outputs_all)
        results_csv.update(hrm_outputs_csv)
        results_all.update(hrm_outputs_all)

        return results_csv, results_all

    def _dynamic_threshold(self, scores, min_nonzero=5) -> float:
        """
        Compute a simple dynamic cutoff τ from positive scores:
        - keep scores > 0
        - IQR-trim outliers
        - τ = mean(trimmed)
        Fallback to 75th percentile if too few points.
        """
        arr = [s for s in scores if s > 0]
        if not arr:
            return 0.0
        if len(arr) < min_nonzero:
            arr_sorted = sorted(arr)
            k = int(0.75 * (len(arr_sorted) - 1))
            return float(arr_sorted[k])

        arr_sorted = sorted(arr)
        q1_idx = int(0.25 * (len(arr_sorted) - 1))
        q3_idx = int(0.75 * (len(arr_sorted) - 1))
        q1, q3 = float(arr_sorted[q1_idx]), float(arr_sorted[q3_idx])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        trimmed = [x for x in arr if (low <= x <= high)]
        if not trimmed:
            trimmed = arr
        return float(sum(trimmed) / len(trimmed))

    def find_common_papers(
        self,
        category_to_papers: dict,
        paper_keyword_map: dict,
        title_map: dict,
        kw_lookup: dict,
        abstract_map: dict,
        threshold: float = 1.5,
    ) -> tuple:
        """
        Returns:
            - static_ranked: list of (paper_id, static_score)
            - dynamic_ranked: list of (paper_id, dynamic_score)
            - low_sorted: list of (paper_id, count) with low relevance (e.g., title fails)
            - df_pairs_file_path: path to saved CSV of shared category pair overlaps
            - pair_counts: dict paper_id -> pair_count (for HRM features)
        """

        # 1) Process user query
        LOG.info(f"🔍 User query: {self.keyword_query}")

        # 2) Classify query
        categories = self.classify_query_with_fallback(self.keyword_query, kw_lookup)

        if self.bypass_filtering:
            categories = [(L, C, next(iter(items[0]["keywords"]), ""))
                          for L, cats in self.ontology.items()
                          for C, items in cats.items()]

        if not categories:
            LOG.error("⚠️ No valid categories found.")
            return [], [], [], "", {}

        LOG.info(f"\n✅ Categories Used: {len(categories)}")
        for i, (layer, cat, kw) in enumerate(categories, 1):
            LOG.info(f"  {i}. '{kw}' → {layer} / {cat}")

        # Expand with opposites/synonyms
        query_keywords = [kw.lower() for _, _, kw in categories]
        opposites = self.ai_tool.get_opposites(query_keywords)
        synonyms = self.ai_tool.get_synonyms(query_keywords)
        expanded_keywords = {kw: [kw] + synonyms.get(kw, []) for kw in query_keywords}
        cat_to_terms = self._build_category_terms(categories, query_keywords, synonyms)

        # Map each (layer, category) to the set of query terms we should count for it
        cat_to_terms = defaultdict(set)
        for (layer, cat, qk) in categories:
            for t in expanded_keywords.get(qk, [qk]):
                t = (t or "").strip().lower()
                if t:
                    cat_to_terms[(layer, cat)].add(t)

        LOG.info("🧪 Opposites used for filtering:")
        for qk in query_keywords:
            LOG.info(f"  - '{qk}': {opposites.get(qk, [])}")

        # ---- Dynamic scorer IDF prep ----
        candidate_terms = set()
        for qk, syns in expanded_keywords.items():
            candidate_terms.add(qk)
            candidate_terms.update(list(syns or []))

        N = len(title_map)
        df = {t: 0 for t in candidate_terms}
        for (_, _), pid2kws in paper_keyword_map.items():
            for pid, kws in pid2kws.items():
                for t in candidate_terms:
                    if t in kws:
                        df[t] += 1

        IDF_MIN, IDF_MAX = 0.0, 3.0
        idf = {}
        for t in candidate_terms:
            n_t = df.get(t, 0)
            raw = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)
            idf[t] = max(IDF_MIN, min(IDF_MAX, raw))

        # ---- Collect per-category hits and scores ----
        per_cat_hits = {}
        static_scores_raw = defaultdict(float)  # M(p) before penalty
        static_scores = defaultdict(float)
        static_matched_terms = defaultdict(set)  # optional: which terms contributed (for debug/export)

        # ---- Collect per-category hits and the corresponding words ---
        dynamic_matched_terms = defaultdict(set)
        static_text_terms = defaultdict(set)
        pids_seen: Set[str] = set()

        paper_keyword_frequencies = defaultdict(lambda: defaultdict(int))
        filtered_out = {}

        # Accumulators for BM25 dynamic scoring
        bm25_tf = defaultdict(lambda: defaultdict(int))  # term frequencies per paper
        doc_len_map: Dict[str, int] = {}  # |p| = tokenized length of title+abstract

        for (layer, cat), hits in category_to_papers.items():
            if (layer, cat) not in [(l, c) for l, c, _ in categories]:
                continue

            per_cat_hits[(layer, cat)] = hits
            terms_for_cat = cat_to_terms.get((layer, cat), set())

            for pid, _count in hits.items():

                pids_seen.add(pid)
                kw_set = paper_keyword_map.get((layer, cat), {}).get(pid, set())

                # --- STATIC (paper-correct): occurrences in title + abstract for this category's query-terms
                t_text = (title_map.get(pid, "") or "").lower()
                a_text = (str(abstract_map.get(pid, "") or "")).lower()

                if pid not in doc_len_map:
                    doc_len_map[pid] = len(re.findall(r'\w+', f"{t_text} {a_text}"))

                occ_total = 0
                matched = set()
                for term in terms_for_cat:
                    if not term:
                        continue
                    ct = len(re.findall(rf'\b{re.escape(term)}\b', t_text))
                    ca = len(re.findall(rf'\b{re.escape(term)}\b', a_text))

                    if ct+ca >0:
                        matched.add(term)
                        if term not in bm25_tf[pid]:
                            bm25_tf[pid][term] = ct+ca
                            static_text_terms[pid].add(term)
                        paper_keyword_frequencies[pid][term] += ct+ca
                    occ_total += ct+ca

                static_scores_raw[pid] += float(occ_total)
                if matched:
                    static_matched_terms[pid].update(matched)

        #--- Compute ontology-filtered BM25 dynamic scores ----

        k1, b = 1.2, 0.75
        # Average document length over candidates we touched
        nonzero_lengths = [dl for dl in doc_len_map.values() if dl > 0]
        avgdl = (sum(nonzero_lengths) / max(1, len(nonzero_lengths))) if nonzero_lengths else 1.0

        # Reset/compute dynamic_scores using BM25 with our precomputed IDF
        dynamic_scores = defaultdict(float)
        for pid in pids_seen:
            dl = doc_len_map.get(pid, 0) or 1
            score = 0.0
            for term, tf in bm25_tf[pid].items():
                if term not in idf:
                    continue
                denom = tf + k1 * (1.0 - b + b * (dl / (avgdl or 1.0)))
                score += idf[term] * ((tf * (k1 + 1.0)) / denom)
            if score > 0:
                dynamic_scores[pid] = float(score)
                dynamic_matched_terms[pid] = set(bm25_tf[pid].keys())

        # ---- Static Scoring ----

        for pid in set(static_scores_raw.keys()):
            t_text = str(title_map.get(pid, ""))
            a_text = str(abstract_map.get(pid, ""))
            rel, opp = self._count_relevant_opposing(
                t_text, a_text, query_keywords, expanded_keywords, opposites
            )
            LOG.info(f"Static: PID {pid}: rel={rel}, opp={opp}")
            LOG.info(f"Penalty for Static: PID {pid} = {self._compute_penalty(opp=opp, rel=rel)}")

            static_scores[pid] = static_scores_raw[pid] * self._compute_penalty(opp=opp, rel=rel)

        static_ranked = []
        for pid, score in static_scores.items():
            if score>=0:
                static_ranked.append((pid, float(score)))

        # ---- Dynamic survivors with τ + opposites ----
        tau_dynamic = 0.0 if self.bypass_filtering else self._dynamic_threshold(list(dynamic_scores.values()))
        LOG.info(f"Dynamic threshold (τ) = {tau_dynamic:.3f}")

        dynamic_ranked = []
        for pid, score in dynamic_scores.items():
            if score <= 0:
                continue

            # --- Apply polarity penalty multiplicatively (same scheme as static) ---
            t_text = str(title_map.get(pid, ""))
            a_text = str(abstract_map.get(pid, ""))
            rel, opp = self._count_relevant_opposing(
                t_text, a_text, query_keywords, expanded_keywords, opposites
            )
            LOG.info(f"Dynamic: PID {pid}: rel={rel}, opp={opp}")
            LOG.info(f"Penalty for Dynamic: PID {pid} = {self._compute_penalty(opp=opp, rel=rel)}")
            score = score * self._compute_penalty(opp=opp, rel=rel)

            if score >= tau_dynamic:
                dynamic_ranked.append((pid, float(score)))

        # Build raw per-keyword (with synonyms) occurrence counts per paper

        occ_detail = {}
        for pid in pids_seen:
            t_text = (title_map.get(pid, "") or "").lower()
            a_text  = (str(abstract_map.get(pid, "") or "")).lower()

            per_kw_counts = {}
            for base_kw, variants in expanded_keywords.items():
                terms = set([base_kw] + (variants + []))
                c_title, c_abs = 0,0
                for term in terms:
                    c_title += len(re.findall(rf'\b{re.escape(term)}\b', t_text))
                    c_abs += len(re.findall(rf'\b{re.escape(term)}\b', a_text))
                per_kw_counts[base_kw] = {"title": int(c_title), "abstract": int(c_abs)}
            occ_detail[pid] = per_kw_counts
        self._occ_detail = occ_detail

        # Persist τ
        try:
            pd.DataFrame([{"tau_dynamic": tau_dynamic}]).to_csv(
                f"{self.output_dir}/dynamic_threshold.csv", index=False
            )
        except Exception:
            pass

        ranked_paper_ids = {pid for pid, _ in set(static_ranked) | set(dynamic_ranked)}

        # ---- Export filtered-out breakdown (optional) ----
        if filtered_out:
            summary_df = PaperRanker.summarize_filtered_papers_with_opposites(
                filtered_out, query_keywords, opposites,
                paper_keyword_frequencies, title_map, abstract_map
            )
            try:
                from IPython.display import display
                display(summary_df)
            except Exception:
                pass
            summary_df.to_csv(f"{self.output_dir}/filtered_paper_breakdown.csv", index=False)
            LOG.info(f"\n🚫 Filtered out {len(filtered_out)} papers due to dominance of opposite keywords.")

        low_relevance = [
            (pid, info.get("title_exceeded", 0))
            for pid, info in filtered_out.items() if info["reason"] == "title_fail"
        ]
        low_sorted = sorted(low_relevance, key=lambda x: x[1])
        # ---- Pair overlaps & file ----
        if len(per_cat_hits) < 2:
            LOG.info("\n⚠️ Overlap analysis needs 2+ categories. Creating fallback pair file.")
            # Fallback: zero pair counts
            pair_counts = {pid: 0 for pid in ranked_paper_ids}
            df_pairs = pd.DataFrame([{"paper_id": pid, "Pair Count": 0, "Shared Pairs": ""} for pid in ranked_paper_ids])
            df_pairs_file_path = f"{self.output_dir}/shared_pair_ranked_papers.csv"
            df_pairs.to_csv(df_pairs_file_path, index=False)
            return static_ranked, dynamic_ranked, low_sorted, df_pairs_file_path, pair_counts

        ranked_by_pair_overlap, total_possible_pairs = PaperRanker.rank_by_pair_overlap_filtered(
            {k: {pid: v for pid, v in d.items() if pid in ranked_paper_ids} for k, d in per_cat_hits.items()},
            ranked_paper_ids
        )

        LOG.info(f"\n🏆 Full Ranking by Number of Category Pairs Shared (out of {total_possible_pairs} pairs):")
        for i, (pid, pair_set) in enumerate(ranked_by_pair_overlap, 1):
            LOG.info(f"  {i}. {pid} → shared in {len(pair_set)}/{total_possible_pairs} pair(s)")

        if not ranked_by_pair_overlap:
            LOG.info("⚠️ No multi-category-pair overlaps found.")

        # Export overlaps
        df_pairs = pd.DataFrame([
            {
                "paper_id": pid,
                "Pair Count": len(pair_set),
                "Shared Pairs": ", ".join(sorted(pair_set))
            }
            for pid, pair_set in ranked_by_pair_overlap
        ])
        df_pairs_file_path = f"{self.output_dir}/shared_pair_ranked_papers.csv"
        df_pairs.to_csv(df_pairs_file_path, index=False)
        LOG.info("✅ Exported shared papers to 'shared_pair_ranked_papers.csv'")

        # Build pair_count dict for HRM
        pair_counts = {}
        for pid, pair_set in ranked_by_pair_overlap:
            pair_counts[pid] = len(pair_set)

        self._static_text_terms = {pid: sorted(static_text_terms.get(pid, set())) for pid in ranked_paper_ids}
        self._ontology_matched_terms = {pid: sorted(dynamic_matched_terms.get(pid, set())) for pid in ranked_paper_ids}

        return static_ranked, dynamic_ranked, low_sorted, df_pairs_file_path, pair_counts

    # ---------------------------
    # CLASSIFICATION & FILTERING
    # ---------------------------

    def classify_query_with_fallback(self, user_query: str, kw_lookup: dict) -> list:
        tokens = [t for t in re.findall(r"\w+", user_query.lower()) if len(t) > 1]

        # summarize ontology for prompt brevity
        ontology_summary = {layer: list(cats.keys()) for layer, cats in self.ontology.items()}

        prompt = f'''
                    You are a structured classification AI for materials science.
                    Ontology:
                    {json.dumps(ontology_summary, indent=2)}
                    Given the user query: "{user_query}"
                    Extract each meaningful keyword (ignore stopwords like 'and', 'for', etc).
                    Assign each keyword to the most relevant category and layer from the ontology.
                    Use both the category name and its associated keywords to guide your assignment.
                    Return only the results in valid JSON format:
                    [
                    {{
                        "keyword": "cheap",
                        "category": "Earth-Abundant Elements",
                        "layer": "Elemental Composition"
                    }},
                    ...
                    ]
                '''

        response = self.ai_tool.ask(prompt).strip().replace("```json", "").replace("```", "")
        try:
            model_output = json.loads(response)
        except json.JSONDecodeError:
            LOG.warning("⚠️ AI output couldn't be parsed. Falling back.")
            model_output = []

        categories = []
        found_set = set()
        model_keywords = set()
        for item in model_output:
            kw = item.get("keyword", "").lower()
            layer = item.get("layer", "Unknown")
            cat = item.get("category", "Unknown")
            if not kw:
                continue
            model_keywords.add(kw)
            if layer != "Unknown" and cat != "Unknown":
                if (layer, cat) not in found_set:
                    categories.append((layer, cat, kw))
                    found_set.add((layer, cat))

        for tok in tokens:
            if tok not in model_keywords and tok in kw_lookup:
                layer, cat = kw_lookup[tok]
                if (layer, cat) not in found_set:
                    categories.append((layer, cat, tok))
                    found_set.add((layer, cat))
        return categories

    def _score_with_bm25(
            self,
            candidate_ids: list,
            title_map: dict,
            abstract_map: dict,
    ):
        """
        BM25 over title+abstract, restricted to candidate_ids so it aligns
        with the existing pair-count file (P(p)).
        Returns: list of (paper_id, score) sorted desc.
        """
        if not BM25_AVAILABLE:
            from sokegraph.util.logger import LOG
            LOG.warning("BM25 not available (rank_bm25 not installed). Skipping BM25 track.")
            return [(pid, 0.0) for pid in candidate_ids]

        import re
        def tok(s: str):
            return re.findall(r"\w+", (s or "").lower())

        # Build corpus in the exact order of candidate_ids
        corpus = []
        for pid in candidate_ids:
            text = f"{title_map.get(pid, '')} {abstract_map.get(pid, '')}"
            corpus.append(tok(text))

        bm25 = BM25Okapi(corpus, k1=1.2, b=0.75)

        # Use the raw user query (you can expand with synonyms if you want)
        q_tokens = tok(self.keyword_query)
        scores = bm25.get_scores(q_tokens)  # one per candidate

        ranked = list(zip(candidate_ids, map(float, scores)))
        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked

    def _score_with_HRM(
            self,
            candidate_ids: List[str],
            title_map: Dict[str, str],
            abstract_map: Dict[str, str],
            paper_keyword_map: Dict[Tuple[str, str], Dict[str, Set[str]]],
            kw_lookup: Dict[str, Tuple[str, str]],
            a: float = 1.0, b: float = 0.5, gamma: float = 0.3, delta: float = 0.3, kappa: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Ontology-first hierarchical reasoning scorer (fallback for HRM).
        - Category evidence combines ontology-keyword overlap and text support.
        - Layer score rewards coverage & consistency.
        - Cross-layer coherence adds a small bonus.
        Returns: list of (paper_id, score) sorted desc for candidate_ids.
        """

        hrs_matched_terms = defaultdict(set)
        layers_scores_pir_pid = defaultdict(dict)

        # ---- PRINTING DEBUGGING ----
        LOG.info("---- HRM Scoring Debugging ----")
        LOG.info(f"Candidate IDs: {candidate_ids}")

        # Recompute categories + candidate terms from the current keyword query
        categories = self.classify_query_with_fallback(self.keyword_query, kw_lookup)
        if not categories:
            return [(pid, 0.0) for pid in candidate_ids]

        # query keywords + synonyms
        query_keywords = [kw.lower() for _, _, kw in categories]
        synonyms = self.ai_tool.get_synonyms(query_keywords) if hasattr(self.ai_tool, "get_synonyms") else {}
        cand_terms: Set[str] = set()
        for q in query_keywords:
            cand_terms.add(q)
            cand_terms.update([s.lower() for s in (synonyms.get(q, []) or [])])

        expanded_keywords = {q: [s.lower() for s in synonyms.get(q, [])] for q in query_keywords}
        opposites = self.ai_tool.get_opposites(query_keywords) if hasattr(self.ai_tool, "get_opposites") else {}

        # device layer helpers
        device_layer_name = "Device"

        # PAIR-SAFE NORMALIZATION: categories may be (l,c) or (l,c,score) ---

        def _to_pair(x):
            # Accept (l,c) or (l,c,*) and return (l,c)
            try:
                return (x[0], x[1])
            except Exception:
                return None

        selected_cats = {p for p in (_to_pair(x) for x in categories) if p}
        q_device = {c for (l, c) in selected_cats if str(l).lower() == device_layer_name.lower()}
        all_device_cats = {cat for (l, cat) in paper_keyword_map.keys() if str(l).lower() == device_layer_name.lower()}

        # --- Modified for strict device weighting: sibling contributions suppressed ---
        alpha_sib = 0.0  # formerly 0.4

        # Precompute text hits per pid, per term
        def _count_occ(text, term) -> int:
            txt = self._str_or_empty(text).lower()
            t = self._str_or_empty(term)
            if not t:
                return 0
            return len(re.findall(rf'\b{re.escape(t)}\b', txt))

        title_hits: Dict[str, Dict[str, int]] = {
            pid: {t: _count_occ(title_map.get(pid, ""), t) for t in cand_terms} for pid in candidate_ids
        }
        abs_hits: Dict[str, Dict[str, int]] = {
            pid: {t: _count_occ(abstract_map.get(pid, ""), t) for t in cand_terms} for pid in candidate_ids
        }

        # Category scores
        CatScore: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        layer_cats_touched: Dict[str, Set[str]] = defaultdict(set)

        # Build per-category ontology keyword sets across papers
        # paper_keyword_map[(layer, cat)][pid] -> {kw,...} already exists
        for (layer, cat), pid2kw in paper_keyword_map.items():
            # Keep only categories present in the classified selection (ontology-first focus)
            is_device_layer = str(layer).lower() == device_layer_name.lower()
            if ((layer, cat) not in selected_cats) and (not is_device_layer):
                continue

            # build allowed set for this layer: if classifier gave none, fall back to all cats in this layer
            if not is_device_layer:
                layer_selected = {c for (l, c) in selected_cats if l == layer}
                if not layer_selected:
                    layer_selected = set(self.ontology.get(layer, {}).keys())
                if cat not in layer_selected:
                    continue

            for pid in candidate_ids:
                kws = pid2kw.get(pid, set())
                onto_match = kws & cand_terms
                if onto_match:
                    hrs_matched_terms[pid].update(onto_match)
                    layer_cats_touched[layer].add(cat)
                    cat_hit = float(len(onto_match))  # ontology evidence
                    text_hit = float(sum(math.log1p(title_hits[pid].get(t, 0) + abs_hits[pid].get(t, 0)) for t in
                                         onto_match))  # normalized_text_support
                    score = a * cat_hit + b * text_hit
                    if score > 0:
                        CatScore[(layer, cat)][pid] = score

        # Layer aggregation: coverage + consistency
        LayScore: Dict[str, Dict[str, float]] = defaultdict(dict)
        for layer, cats in self.ontology.items():
            # relevant categories for this layer (from the classified information)
            # for device
            if str(layer).lower() == device_layer_name.lower():
                rel_cats = ({c for (l, c) in selected_cats if l == layer} | all_device_cats)
            else:
                rel_cats = {c for (l, c) in selected_cats if l == layer}
                if not rel_cats:
                    rel_cats = {c for (l, c2) in paper_keyword_map.keys() if l == layer for c in [c2]}

            # skip layers not touched by the query
            if not rel_cats:
                continue

            for pid in candidate_ids:

                if str(layer).lower() == device_layer_name.lower():
                    exact_sum = sum(CatScore.get((layer, c), {}).get(pid, 0.0) for c in q_device)
                    sib_cats = [c for c in rel_cats if c not in q_device]
                    sib_sum = sum(CatScore.get((layer, c), {}).get(pid, 0.0) for c in sib_cats)
                    lay_raw = exact_sum + alpha_sib * sib_sum

                    # active per-category scores (for coverage/consistency)
                    cat_scores = [CatScore.get((layer, c), {}).get(pid, 0.0) for c in rel_cats if
                                  pid in CatScore.get((layer, c), {})]

                else:
                    cat_scores = [CatScore.get((layer, c), {}).get(pid, 0.0)
                                  for c in rel_cats if pid in CatScore.get((layer, c), {})]
                    lay_raw = sum(cat_scores)

                if not cat_scores:
                    continue

                # coverage: fraction of relevant categories activated for this paper
                lay_cov = len(cat_scores) / max(1, len(rel_cats))

                # consistency: normalized entropy across active categories
                total = sum(cat_scores)
                if total > 0 and len(cat_scores) >= 2:
                    probs = [s / total for s in cat_scores]
                    H = -sum(p * math.log(p + 1e-12) for p in probs)
                    cons_norm = H / math.log(float(len(cat_scores)))
                else:
                    cons_norm = 0.0

                layer_weight = self._get_layer_weight(layer)
                LayScore[layer][pid] = lay_raw * (1.0 + gamma * lay_cov + delta * cons_norm) * layer_weight
                layers_scores_pir_pid[layer][pid] = LayScore[layer][pid]

        # Cross-layer coherence bonus
        paper_layer_presence: Dict[str, Set[str]] = defaultdict(set)
        for (layer, cat), m in CatScore.items():
            for pid in m.keys():
                paper_layer_presence[pid].add(layer)

        scores: Dict[str, float] = {}
        for pid in candidate_ids:
            base = sum(LayScore[layer].get(pid, 0.0) for layer in self.ontology.keys())
            layers = sorted(list(paper_layer_presence.get(pid, set())))
            bonus = kappa * float(len(layers) ** 2)

            # Apply polarity penalty with new formulation
            t_text = self._str_or_empty(title_map.get(pid, ""))
            a_text = self._str_or_empty(abstract_map.get(pid, ""))
            rel, opp = self._count_relevant_opposing(t_text, a_text, query_keywords, expanded_keywords, opposites)
            LOG.info(f"HRM: PID {pid}: rel={rel}, opp={opp}")
            LOG.info(f"Penalty for HRM: PID {pid} = {self._compute_penalty(opp=opp, rel=rel)}")
            scores[pid] = (base + bonus) * self._compute_penalty(opp=opp, rel=rel)

        ranked = sorted(scores.items(), key=lambda t: t[1], reverse=True)

        self._debug_matches["hrm"] = {str(pid): sorted(hrs_matched_terms.get(pid, set()))
                                      for pid, _ in ranked}
        self._hrs_layer_scores = {pid: d for pid, d in layers_scores_pir_pid.items()}

        return ranked

    def _score_with_simple_hrm_fallback(
        self,
        candidate_ids: List[str],
        title_map: Dict[str, str],
        abstract_map: Dict[str, str],
        paper_keyword_map: Dict[Tuple[str, str], Dict[str, Set[str]]],
        kw_lookup: Dict[str, Tuple[str, str]],
        a: float = 1.0,
        b: float = 0.5,
        gamma: float = 0.3,
        delta: float = 0.3,
        kappa: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Simple HRM fallback from paper_ranker.py.txt.
        This is a cleaner, more straightforward implementation without device layer complexity.
        """
        categories = self.classify_query_with_fallback(self.keyword_query, kw_lookup)
        if not categories:
            return [(pid, 0.0) for pid in candidate_ids]
        query_keywords = [kw.lower() for _, _, kw in categories]
        synonyms = getattr(self.ai_tool, "get_synonyms", lambda x: {})(query_keywords) or {}
        opposites = getattr(self.ai_tool, "get_opposites", lambda x: {})(query_keywords) or {}

        cand_terms: Set[str] = set()
        for q in query_keywords:
            cand_terms.add(q)
            cand_terms.update([s.lower() for s in (synonyms.get(q, []) or [])])

        CatScore: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        for (layer, cat), pid2kw in paper_keyword_map.items():
            for pid in candidate_ids:
                kws = pid2kw.get(pid, set())
                onto = kws & cand_terms
                if not onto:
                    continue
                t_text = (title_map.get(pid, "") or "").lower()
                a_text = (abstract_map.get(pid, "") or "").lower()
                text_hit = 0.0
                for t in onto:
                    text_hit += math.log1p(len(re.findall(rf'\b{re.escape(t)}\b', t_text)) + len(re.findall(rf'\b{re.escape(t)}\b', a_text)))
                score = a * float(len(onto)) + b * text_hit
                if score > 0:
                    CatScore[(layer, cat)][pid] = score

        LayScore: Dict[str, Dict[str, float]] = defaultdict(dict)
        for layer in self.ontology.keys():
            rel_cats = {c for (l, c, _kw) in categories if l == layer}
            if not rel_cats:
                rel_cats = {c for (l, c) in [k for k in paper_keyword_map.keys() if k[0] == layer]}
            if not rel_cats:
                continue
            for pid in candidate_ids:
                cat_scores = [CatScore.get((layer, c), {}).get(pid, 0.0) for c in rel_cats if pid in CatScore.get((layer, c), {})]
                if not cat_scores:
                    continue
                lay_raw = sum(cat_scores)
                lay_cov = len(cat_scores) / max(1, len(rel_cats))
                if len(cat_scores) >= 2 and sum(cat_scores) > 0:
                    probs = [s / sum(cat_scores) for s in cat_scores]
                    H = -sum(p * math.log(p + 1e-12) for p in probs)
                    cons_norm = H / math.log(float(len(cat_scores)))
                else:
                    cons_norm = 0.0
                layer_weight = self._get_layer_weight(layer)
                LayScore[layer][pid] = lay_raw * (1.0 + gamma * lay_cov + delta * cons_norm) * layer_weight

        paper_layers: Dict[str, Set[str]] = defaultdict(set)
        for (layer, cat), d in CatScore.items():
            for pid in d.keys():
                paper_layers[pid].add(layer)

        scores: Dict[str, float] = {}
        for pid in candidate_ids:
            base = sum(LayScore[layer].get(pid, 0.0) for layer in self.ontology.keys())
            bonus = kappa * float(len(paper_layers.get(pid, set())) ** 2)
            rel, opp = self._count_relevant_opposing((title_map.get(pid, "") or ""), (abstract_map.get(pid, "") or ""), query_keywords, {q: [q] + (synonyms.get(q, []) or []) for q in query_keywords}, opposites)
            scores[pid] = (base + bonus) * self._simple_penalty(rel, opp)

        ranked = sorted(scores.items(), key=lambda t: t[1], reverse=True)
        return ranked

    def _simple_penalty(self, rel: int, opp: int) -> float:
        """
        Simple penalty calculation from paper_ranker.py.txt.
        """
        denom = rel + opp + 1.0
        return 1.0 - self.polarity_lambda * (opp / denom)

    def _compute_penalty(self, opp: int, rel: int, penalize: bool = False):
        if penalize: 
            denom = float(rel) + float(opp) + 1e-9
            penalty = self.static_lambda * (float(opp) / denom)
            return max(0.0, 1.0-penalty)
        return 1.0

    def _get_layer_weight(self, layer: str) -> float:
        """Get priority weight for a given ontology layer.
        Returns configured weight or 1.0 (neutral) for unlisted layers.
        """
        weight = self.layer_priority_weights.get(layer, 1.0)
        if weight != 1.0:
            LOG.debug(f"Layer '{layer}' priority weight: {weight}x")
        return weight

    def _write_comparison_table(
            self,
            candidate_ids: List[str],
            static_ranked: List[Tuple[str, float]],
            dynamic_ranked: List[Tuple[str, float]],
            hrs_ranked: List[Tuple[str, float]],
            output_path: str,
            epsilon: float = 1e-9,
    ):
        """Write a side-by-side comparison table with dense ranks.
           Primary key: Pair Count (desc). Secondary: method score (desc).
           This matches the paper’s rule: P(p) first, then the method score."""
        import numpy as np
        import pandas as pd
        import os

        # ---- Maps for scores
        s_map = {str(pid): float(s) for pid, s in static_ranked}
        d_map = {str(pid): float(s) for pid, s in dynamic_ranked}
        h_map = {str(pid): float(s) for pid, s in hrs_ranked}

        # ---- Load pair counts (force IDs to str for consistent joins)
        pair_counts: Dict[str, int] = {}
        try:
            df_pairs = pd.read_csv(os.path.join(self.output_dir, "shared_pair_ranked_papers.csv"))
            pair_counts = {
                str(pid): int(pc)
                for pid, pc in zip(df_pairs["paper_id"].astype(str), df_pairs["Pair Count"].astype(int))
            }
        except Exception:
            pair_counts = {}

        # ---- Candidate ID order (force str)
        base_order = [str(pid) for pid in candidate_ids]
        title_map = self._get_title_map()
        abstract_map = self._get_abstract_map()

        def _short(txt, n=280):
            t = (txt or "")
            return t if len(t) <= n else t[:n].rstrip() + "…"

        # ---- Dense rank function: Pair Count desc, then Score desc
        def _dense_rank_pair_first(score_map: Dict[str, float], order: List[str]) -> Dict[str, int]:
            pos = {pid: i for i, pid in enumerate(order)}  # stable fallback
            items = []
            for pid in order:
                pc = float(pair_counts.get(pid, 0))
                sc = float(score_map.get(pid, 0.0))
                items.append((pid, pc, sc))
            # Sort: higher Pair Count first, then higher score
            items.sort(key=lambda t: (-t[1], -t[2], pos[t[0]]))

            ranks: Dict[str, int] = {}
            prev_pc, prev_sc = None, None
            current_rank = 0
            for pid, pc, sc in items:
                if prev_pc is None or pc != prev_pc or abs(sc - prev_sc) > epsilon:
                    current_rank += 1
                    prev_pc, prev_sc = pc, sc
                ranks[pid] = current_rank
            return ranks

        # ---- Robust min–max normalization (0–1)

        def _min_max_scale(values, eps=1e-9) -> pd.Series:
            """
            Robust 0–1 scaling based on percent ranks (dense). Preserves ordering,
            ties share the same value. Safe with outliers and constant arrays.
            """
            s = pd.Series(values, dtype="float64")
            if s.empty:
                return s
            # dense percent rank in [0,1]
            vmin, vmax = float(s.min()), float(s.max())
            return (s-vmin)/(vmax-vmin + eps)

        # ---- Compute ranks (pair-first) for each method
        r_s = _dense_rank_pair_first(s_map, base_order)
        r_d = _dense_rank_pair_first(d_map, base_order)
        r_h = _dense_rank_pair_first(h_map, base_order)

        # ---- Build table (now includes Pair Count for transparency)
        df = pd.DataFrame({
            "paper": base_order,
            "title": [title_map.get(pid, "") for pid in base_order],
            "abstract": [(abstract_map.get(pid, "")) for pid in base_order],
            "pair_count": [pair_counts.get(pid, 0) for pid in base_order],
            "score_static": [s_map.get(pid, 0.0) for pid in base_order],
            "score_dynamic": [d_map.get(pid, 0.0) for pid in base_order],
            "score_HRM": [h_map.get(pid, 0.0) for pid in base_order],
            "rank_static": [r_s[pid] for pid in base_order],
            "rank_dynamic": [r_d[pid] for pid in base_order],
            "rank_HRM": [r_h[pid] for pid in base_order],
            "static_text_terms": [
                ", ".join(getattr(self, "_static_text_terms", {}).get(pid, [])) for pid in base_order
            ],
            "ontology_matched_terms (dynamic)": [
                ", ".join(getattr(self, "_ontology_matched_terms", {}).get(pid, [])) for pid in base_order
            ]
        })

        # --- HRS per-layer breakdown ---
        hrs_layers = getattr(self, "_hrs_layer_scores", {}) or {}  # {pid -> {layer -> score}}

        def _layer_str(d: dict) -> str:
            if not d:
                return ""
            return "; ".join(f"{str(k)}:{float(v):.3f}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0])))

        # Compact string column per paper
        df["HRM_layer_scores"] = [
            _layer_str(hrs_layers.get(pid, {})) for pid in base_order
        ]

        # One sortable numeric column per ontology layer
        for layer in getattr(self, "ontology", {}).keys():
            df[f"hrs_layer_{str(layer)}"] = [
                float(hrs_layers.get(pid, {}).get(layer, 0.0)) for pid in base_order
            ]
        # --- end HRS per-layer breakdown ---

        # Normalized scores (for easy cross-method comparison; ranks are unaffected)
        df["score_static_norm"] = _min_max_scale(df["score_static"])
        df["score_dynamic_norm"] = _min_max_scale(df["score_dynamic"])
        df["score_HRM_norm"] = _min_max_scale(df["score_HRM"])
        occ = getattr(self, "_occ_detail", {}) or {}
        df["occurrence_breakdown"] = [
            json.dumps(occ.get(pid, {}), ensure_ascii=False) for pid in df["paper"]
        ]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        # --- Also export a colored Excel version for readability ---
        try:
            xlsx_path = (
                output_path[:-4] + ".xlsx" if output_path.lower().endswith(".csv")
                else output_path + ".xlsx"
            )

            # Write to Excel with XlsxWriter so we can apply conditional formatting
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                sheet = "comparison"
                df.to_excel(writer, index=False, sheet_name=sheet)

                workbook  = writer.book
                worksheet = writer.sheets[sheet]

                # Aesthetics
                worksheet.freeze_panes(1, 0)  # freeze header
                worksheet.autofilter(0, 0, 0, len(df.columns) - 1)

                # Widen some useful columns if present
                def _safe_set_width(colname, width):
                    if colname in df.columns:
                        c = df.columns.get_loc(colname)
                        worksheet.set_column(c, c, width)
                _safe_set_width("paper", 22)
                _safe_set_width("title", 50)
                _safe_set_width("matched_static_terms", 28)
                _safe_set_width("matched_dynamic_terms", 28)

                # Formats for traffic-light coloring
                green  = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
                yellow = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})
                red    = workbook.add_format({"bg_color": "#F2DCDB", "font_color": "#9C0006"})

                # Helper to color a column by normalized thresholds
                def color_norm_column(colname):
                    if colname not in df.columns:
                        return
                    col_idx = df.columns.get_loc(colname)
                    # Data rows start at row 1 (row 0 is header)
                    first_row, last_row = 1, len(df)
                    # Green: >= 0.70
                    worksheet.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "cell", "criteria": ">=", "value": 0.70, "format": green
                    })
                    # Yellow: 0.40 to < 0.70
                    worksheet.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "cell", "criteria": "between", "minimum": 0.40, "maximum": 0.6999999, "format": yellow
                    })
                    # Red: < 0.40
                    worksheet.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "cell", "criteria": "<", "value": 0.40, "format": red
                    })

                # Apply to normalized score columns (created earlier in this function)
                color_norm_column("score_static_norm")
                color_norm_column("score_dynamic_norm")
                color_norm_column("score_HRM_norm")

            LOG.info(f"📗 Saved colored Excel comparison to: {xlsx_path}")
        except Exception as e:
            LOG.warning(f"Could not write colored Excel comparison: {e}")

    @staticmethod
    def summarize_filtered_papers_with_opposites(
        filtered_out: Dict[str, Dict[str, Any]],
        query_keywords: List[str],
        opposites: Dict[str, List[str]],
        paper_keyword_frequencies: Dict[str, Dict[str, int]],
        title_map: Dict[str, str],
        abstract_map: Dict[str, str],
    ) -> pd.DataFrame:
        rows = []
        for pid, info in filtered_out.items():
            title_text = (title_map.get(pid, "") or "").lower()
            abstract_text = str((abstract_map.get(pid, "") or "")).lower()
            for qk in query_keywords:
                matched_opp_keywords = []
                title_rel = len(re.findall(rf'\b{re.escape(qk)}\b', title_text))
                title_opp = 0
                for opp in opposites.get(qk, []):
                    count = len(re.findall(rf'\b{re.escape(opp)}\b', title_text))
                    title_opp += count
                    if count > 0:
                        matched_opp_keywords.append(opp)
                abs_rel = len(re.findall(rf'\b{re.escape(qk)}\b', abstract_text))
                abs_opp = 0
                for opp in opposites.get(qk, []):
                    count = len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text))
                    abs_opp += count
                    if count > 0 and opp not in matched_opp_keywords:
                        matched_opp_keywords.append(opp)
                total_rel = title_rel + abs_rel
                total_opp = title_opp + abs_opp
                ratio = round(total_opp / total_rel, 2) if total_rel else float("inf")
                status = "Filtered" if ratio > info["threshold"] else "Kept"
                rows.append(
                    {
                        "paper_id": pid,
                        "Query Keyword": qk,
                        "Title Relevant Count": title_rel,
                        "Title Opposing Count": title_opp,
                        "Abstract Relevant Count": abs_rel,
                        "Abstract Opposing Count": abs_opp,
                        "Total Relevant Count": total_rel,
                        "Total Opposing Count": total_opp,
                        "Matched Opposing Keywords": ", ".join(sorted(set(matched_opp_keywords))),
                        "Ratio": ratio,
                        "Status": status,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def rank_by_pair_overlap_filtered(
        per_cat_hits: Dict[Tuple[str, str], Dict[str, float]],
        ranked_paper_ids: Set[str]
    ) -> Tuple[List[Tuple[str, Set[str]]], int]:
        pair_paper_map = defaultdict(set)
        category_pairs = list(combinations(per_cat_hits.keys(), 2))
        total_possible_pairs = len(category_pairs)
        for (cat1, cat2) in category_pairs:
            papers1 = set(per_cat_hits[cat1].keys())
            papers2 = set(per_cat_hits[cat2].keys())
            filtered_shared = (papers1 & papers2) & ranked_paper_ids
            pair_label = f"{cat1[1]} ↔ {cat2[1]}"
            for pid in filtered_shared:
                pair_paper_map[pid].add(pair_label)
        ranked_by_pair_overlap = sorted(
            pair_paper_map.items(),
            key=lambda x: len(x[1]),  # sort by number of pairs
            reverse=True
        )
        return ranked_by_pair_overlap, total_possible_pairs

    @staticmethod
    def _str_or_empty(x):
        if isinstance(x, str):
            return x
        if x is None:
            return ""
        try:
            import pandas as pd  # already imported above
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x)  # last resort

    def rank_shared_papers_by_pairs_and_mentions(self,
        ranked: List[Tuple[str, float]],
        low_sorted: List[Tuple[str, float]],
        pair_file: str,
        output_dir: str,
        label: str = "static"
    ) -> Dict[str, str]:
        try:
            df_pairs = pd.read_csv(pair_file)
        except FileNotFoundError:
            LOG.error(f"❌ File not found: {pair_file}")
            return {}, {}
        except pd.errors.EmptyDataError:
            LOG.warning(f"⚠️ Empty CSV file: {pair_file}")
            return {}, {}

        relevance_scores = {paper_id: ("high", score) for paper_id, score in ranked}
        for paper_id, _ in low_sorted:
            if paper_id not in relevance_scores:
                relevance_scores[paper_id] = ("low", 0)

        df_pairs["Relevance Level"] = df_pairs["paper_id"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[0])
        df_pairs["Relevant Keyword Score"] = df_pairs["paper_id"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[1])
        df_pairs["Scoring"] = label

        df_pairs = df_pairs.sort_values(
            by=["Pair Count", "Relevant Keyword Score"],
            ascending=[False, False]
        ).reset_index(drop=True)

        LOG.info(f"\n🏆 Shared Papers Ranked by Pair Count → Mentions ({label.upper()}):")
        for _, row in df_pairs.iterrows():
            LOG.info(f"{row['paper_id']} | Pairs: {row['Pair Count']} | Score: {row['Relevant Keyword Score']} | Relevance: {row['Relevance Level']}")

        # Merge df_pairs with metadata_df on title (or DOI if available)
        papers_df = pd.DataFrame(self.papers)

        df_merged = df_pairs.merge(
            papers_df,
            on="paper_id",      # <-- change to "doi" if DOI is your unique key
            how="left"
        )


        output_paths_all = {}
        output_csv_paths = {}
        for level in ["high", "low", "unknown"]:
            subset = df_merged[df_merged["Relevance Level"] == level]
            if not subset.empty:
                basename = f"shared_ranked_by_pairs_then_mentions_{label}_{level}"
                paths = PaperRanker.save_papers(subset, self.output_dir, basename)
                output_paths_all[label+"_"+level] = paths
                output_csv_paths[label+"_"+level] = paths["csv"]  # only CSV
                LOG.info(f"✅ Saved CSV: {paths['csv']}")

        return output_csv_paths, output_paths_all

    def _get_title_map(self):
        """
        Map: safe title or paper_id -> title text
        """
        title_map = {}
        for _, row in self.papers.iterrows():
            safe_id = str(row.get("paper_id", ""))
            # Store the abstract (or empty string if missing) as the value
            title_map[safe_id] = self._str_or_empty(row.get("title", ""))
        return title_map


    def _get_abstract_map(self):
        """
         Map: safe title or paper_id -> abstract text
        """
        abstract_map = {}
        for _, row in self.papers.iterrows():
            safe_id = str(row.get("paper_id", ""))
            # Assign abstract text (or an empty string if not available)
            abstract_map[safe_id] = self._str_or_empty( row.get("abstract", ""))
        return abstract_map

    @staticmethod
    def save_papers(df: pd.DataFrame, output_dir: str, basename: str) -> dict:
        """
        Save the DataFrame in multiple formats:
        CSV, JSON, JSONL, JSON-LD, Parquet, RDF/Turtle, GraphML (Neo4j compatible)
        Returns dict of {format: file_path}.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # CSV
        csv_path = os.path.join(output_dir, f"{basename}.csv")
        df.to_csv(csv_path, index=False)
        paths["csv"] = csv_path

        # JSON
        json_path = os.path.join(output_dir, f"{basename}.json")
        df.to_json(json_path, orient="records", indent=2)
        paths["json"] = json_path

        # JSONL
        jsonl_path = os.path.join(output_dir, f"{basename}.jsonl")
        df.to_json(jsonl_path, orient="records", lines=True)
        paths["jsonl"] = jsonl_path

        # JSON-LD (simplified)
        jsonld_path = os.path.join(output_dir, f"{basename}.jsonld")
        context = {"paper_id": "http://example.org/paper_id",
                   "title": "http://purl.org/dc/elements/1.1/title",
                   "doi": "http://purl.org/ontology/bibo/doi"}
        jsonld_data = {"@context": context, "@graph": df.to_dict(orient="records")}
        with open(jsonld_path, "w", encoding="utf-8") as f:
            json.dump(jsonld_data, f, indent=2)
        paths["jsonld"] = jsonld_path

        # Parquet
        parquet_path = os.path.join(output_dir, f"{basename}.parquet")
        df.to_parquet(parquet_path, index=False)
        paths["parquet"] = parquet_path

        # GraphML (Neo4j compatible)
        if "paper_id" in df.columns:
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_node(row["paper_id"], **row.to_dict())
            graphml_path = os.path.join(output_dir, f"{basename}.graphml")
            nx.write_graphml(G, graphml_path)
            paths["graphml"] = graphml_path

        return paths

    def _prepare_neural_hrm_runtime(self):
        """
        Loads/initializes vocab and HRM model (if PyTorch available).
        If weights exist, loads them. If labels exist (and no weights),
        trains a small model and saves it to output_dir/hrm.pt.
        """
        assert TORCH_AVAILABLE, "PyTorch not available"

        # seed
        random.seed(self.hrm_seed)
        torch.manual_seed(self.hrm_seed)

        # device
        self._hrm_device = "cuda" if torch.cuda.is_available() else "cpu"

        # vocab
        if self.hrm_vocab_path and os.path.exists(self.hrm_vocab_path):
            with open(self.hrm_vocab_path, "r", encoding="utf-8") as f:
                self._hrm_vocab = json.load(f)
        else:
            # Build a minimal vocab on the fly from ontology categories + meta tokens
            self._hrm_vocab = self._build_min_vocab_from_ontology()
            # Optionally save
            try:
                path = os.path.join(self.output_dir, "hrm_vocab.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._hrm_vocab, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # model
        self._hrm_model = _NeuralHRM(
            vocab_size=len(self._hrm_vocab),
            dim=self.hrm_dim,
            max_steps=self.hrm_halting_max_steps
        ).to(self._hrm_device)

        # Weights?
        weights_path = self.hrm_weights_path or os.path.join(self.output_dir, "hrm.pt")
        if os.path.exists(weights_path):
            self._hrm_model.load_state_dict(torch.load(weights_path, map_location=self._hrm_device))
            self._hrm_model.eval()
            LOG.info(f"Loaded HRM weights from {weights_path}")
            return

        # Train if labels provided
        if self.hrm_labels_path and os.path.exists(self.hrm_labels_path):
            LOG.info("Training Neural HRM from labels...")
            try:
                self._train_neural_hrm(self.hrm_labels_path, weights_path)
                LOG.info("✅ HRM training complete.")
            except Exception as e:
                LOG.warning(f"⚠️ HRM training failed: {e}. Falling back to linear HRM.")
                self._hrm_model = None
        else:
            LOG.info("No HRM weights or labels provided. Neural HRM disabled; fallback will be used.")
            self._hrm_model = None

    def _build_min_vocab_from_ontology(self) -> Dict[str, int]:
        """
        Minimal vocab: ontology categories (layer/category), plus a few meta tokens.
        """
        vocab = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2}
        idx = 3
        for layer, cats in self.ontology.items():
            layer_tok = f"LAYER::{layer}"
            if layer_tok not in vocab:
                vocab[layer_tok] = idx; idx += 1
            for cat in cats.keys():
                cat_tok = f"CATEGORY::{cat}"
                if cat_tok not in vocab:
                    vocab[cat_tok] = idx; idx += 1
        # Meta tokens (you can extend to match your text file)
        for meta in [
            "META::RECENT", "META::OLD",
            "META::JOURNAL_HIGH", "META::JOURNAL_LOW",
            "META::SEM_OVERLAP_HIGH", "META::SEM_OVERLAP_LOW",
            "META::CONFLICT"
        ]:
            if meta not in vocab:
                vocab[meta] = idx; idx += 1
        return vocab

    def _tokenize_paper(
        self,
        pid: str,
        title_map: Dict[str, str],
        abstract_map: Dict[str, str],
        paper_keyword_map: Dict[Tuple[str, str], Dict[str, Set[str]]],
        pair_counts: Dict[str, int]
    ) -> List[int]:
        """
        Build a token sequence approximating the logic in the text HRM:
        - layer/category tokens observed for this paper
        - metadata tokens (dummy heuristics here; extend with real signals)
        - conflict token if needed (heuristic)
        """
        toks: List[int] = []
        V = self._hrm_vocab
        PAD = V.get("[PAD]", 0)
        SEP = V.get("[SEP]", 2)

        # Ontology tokens: categories this paper matches
        categories_seen = set()
        for (layer, cat), pid2kw in paper_keyword_map.items():
            if pid in pid2kw:
                categories_seen.add((layer, cat))
        for layer, cat in sorted(categories_seen):
            layer_tok = V.get(f"LAYER::{layer}", V["[UNK]"])
            cat_tok = V.get(f"CATEGORY::{cat}", V["[UNK]"])
            toks.extend([layer_tok, cat_tok, SEP])

        # Metadata tokens (heuristics; replace with your real signals if available)
        # Recency: naive heuristic from title length (placeholder)
        title_len = len((title_map.get(pid, "") or ""))
        if title_len > 60:
            toks.append(V.get("META::RECENT", V["[UNK]"]))
        else:
            toks.append(V.get("META::OLD", V["[UNK]"]))
        toks.append(SEP)

        # Journal quality (placeholder)
        # If title contains "Nature" or "Science", mark as high
        title_text = (title_map.get(pid, "") or "")
        if "nature" in title_text.lower() or "science" in title_text.lower():
            toks.append(V.get("META::JOURNAL_HIGH", V["[UNK]"]))
        else:
            toks.append(V.get("META::JOURNAL_LOW", V["[UNK]"]))
        toks.append(SEP)

        # Semantic overlap heuristic: pair_count as proxy
        pc = pair_counts.get(pid, 0)
        if pc >= 2:
            toks.append(V.get("META::SEM_OVERLAP_HIGH", V["[UNK]"]))
        else:
            toks.append(V.get("META::SEM_OVERLAP_LOW", V["[UNK]"]))
        toks.append(SEP)

        # Conflict token heuristic: if title mentions "noble"
        if "noble" in title_text.lower():
            toks.append(V.get("META::CONFLICT", V["[UNK]"]))
            toks.append(SEP)

        # Pad/trim
        if len(toks) < self.hrm_max_seq_len:
            toks += [PAD] * (self.hrm_max_seq_len - len(toks))
        else:
            toks = toks[: self.hrm_max_seq_len]
        return toks

    def _score_with_neural_hrm(
        self,
        static_ranked: List[Tuple[str, float]],
        dynamic_ranked: List[Tuple[str, float]],
        pair_counts: Dict[str, int],
        title_map: Dict[str, str],
        abstract_map: Dict[str, str],
        paper_keyword_map: Dict[Tuple[str, str], Dict[str, Set[str]]],
    ) -> List[Tuple[str, float]]:
        """
        Build tokens for all candidate paper IDs (union of static/dynamic survivors),
        run them through the Neural HRM, and return (pid, score).
        """
        assert self._hrm_model is not None, "Neural HRM model not initialized"
        self._hrm_model.eval()

        candidate_ids = sorted({pid for pid, _ in static_ranked} | {pid for pid, _ in dynamic_ranked})
        X = []
        idx_map = {}
        for i, pid in enumerate(candidate_ids):
            toks = self._tokenize_paper(pid, title_map, abstract_map, paper_keyword_map, pair_counts)
            X.append(toks)
            idx_map[i] = pid

        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.long, device=self._hrm_device)
            # Use paper-style cycles
            scores = self._hrm_model.forward(
                x, N=self.hrm_N, T=self.hrm_T
            ).detach().cpu().numpy().tolist()

        hrm_ranked = [(idx_map[i], float(s)) for i, s in enumerate(scores)]
        # sort descending by score
        hrm_ranked.sort(key=lambda t: t[1], reverse=True)
        return hrm_ranked

    def _train_neural_hrm(self, labels_csv_path: str, save_path: str):
        assert TORCH_AVAILABLE and self._hrm_model is not None
        dataset = _HRMDataset(labels_csv_path)
        if len(dataset) == 0:
            raise RuntimeError("No HRM labels found.")

        self._hrm_model.train()
        opt = torch.optim.AdamW(self._hrm_model.parameters(), lr=self.hrm_lr)

        loader = DataLoader(dataset, batch_size=self.hrm_batch_size,
                            shuffle=True, drop_last=False)

        # z-state carried across segments; detached between segments (deep supervision)
        def _init_state(B, D, dev):
            zL = torch.zeros(1, B, D, device=dev)
            zH = torch.zeros(1, B, D, device=dev)
            return zL, zH

        for epoch in range(self.hrm_train_epochs):
            total = 0.0
            for batch in loader:
                opt.zero_grad()

                # Convert labels to tiny token ids to re-use embedding (no text here)
                # We just make a 1-token input per pid (hash -> token id)
                if dataset.mode == "pair":
                    pid_pos, pid_neg = batch
                    pid_all = list(pid_pos) + list(pid_neg)
                    y_all = [1] * len(pid_pos) + [0] * len(pid_neg)
                else:
                    pid_all, y_all = batch

                # Tokenize: reuse vocab hash to make stable ints
                tok_ids = [self._hrm_vocab.get(f"PID::{p}", self._hrm_vocab.get("[UNK]", 1))
                           for p in pid_all]
                x_tok = torch.tensor(tok_ids, dtype=torch.long, device=self._hrm_device).unsqueeze(1)  # [B,1]
                y = torch.tensor(y_all, dtype=torch.float32, device=self._hrm_device)  # [B]

                B = x_tok.size(0)
                D = self.hrm_dim
                zL, zH = _init_state(B, D, self._hrm_device)

                # Deep supervision with M segments; ACT variant picks halt/continue via q-head
                Mmax = self.hrm_Mmax
                Mmin = 1 if random.random() > self.hrm_eps_minseg else random.randint(2, Mmax)

                seg_losses = []
                for m in range(Mmax):
                    (zL, zH), s, q = self._hrm_model.forward_once(
                        x_tok, zL, zH, self.hrm_N, self.hrm_T
                    )

                    # main loss (pointwise): BCE with logits on score
                    loss_main = F.binary_cross_entropy_with_logits(s, y)

                    # optional Q-learning head for ACT
                    loss_q = 0.0
                    if self._hrm_model.use_act and q is not None:
                        with torch.no_grad():
                            # simple bootstrapped target:
                            # if we are at final segment, target = [1 if correct else 0, 0]
                            pred_correct = (torch.sigmoid(s) > 0.5).float()
                            G_halt = pred_correct
                            # continue target bootstraps from next segment "halt" (here: current pred as proxy)
                            G_cont = torch.max(torch.sigmoid(q), dim=1)[0].detach()
                            G = torch.stack([G_halt, G_cont], dim=1)  # [B,2] in [0,1]
                        loss_q = F.binary_cross_entropy_with_logits(q, G)

                    seg_losses.append(loss_main + loss_q)

                    # Decide to halt (train-time ACT): allow halt only after Mmin
                    if self._hrm_model.use_act and q is not None and m + 1 >= Mmin:
                        probs = torch.softmax(q, dim=1)  # [halt, continue]
                        halt_mask = (probs[:, 0] > probs[:, 1])
                        if halt_mask.all():
                            break

                    # deep supervision: DETACH state before next segment
                    zL = zL.detach()
                    zH = zH.detach()

                loss = torch.stack(seg_losses).mean()
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu().item())

            LOG.info(f"[HRM] epoch {epoch + 1}/{self.hrm_train_epochs} loss={total:.4f}")

        torch.save(self._hrm_model.state_dict(), save_path)
        self._hrm_model.eval()


# ===========================
#   Torch: HRM architecture
# ===========================

class _NeuralHRM(nn.Module):
    """
    Paper-style HRM scorer:
      - Token embedding
      - Two recurrent modules: L (fast) and H (slow)
      - Run N cycles; in each cycle run T L-steps then 1 H-step
      - Output linear head over H; optional ACT Q-head for halting
    """
    def __init__(self, vocab_size: int, dim: int = 128, max_steps: int = 3,
                 use_act: bool = True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)

        # Replace the single GRU with two levels
        self.l_rnn = nn.GRU(input_size=dim*2, hidden_size=dim, batch_first=True)
        self.h_rnn = nn.GRU(input_size=dim*2, hidden_size=dim, batch_first=True)

        # Heads
        self.scorer = nn.Linear(dim, 1)     # relevance logit from H-state
        self.q_head = nn.Linear(dim, 2) if use_act else None  # [halt, continue]

        # Light norms (keeps training stable)
        self.norm_l = nn.LayerNorm(dim)
        self.norm_h = nn.LayerNorm(dim)

        self.use_act = use_act

    def forward_once(self, x_tok: torch.LongTensor,
                     zL: torch.Tensor, zH: torch.Tensor,
                     N: int, T: int):
        """
        A single multi-cycle forward without ACT segmentation.
        x_tok: [B, Ttok]
        zL, zH: initial states [1, B, D]
        returns: final (zL, zH), score [B], optional q_logits [B,2]
        """
        B = x_tok.size(0)
        e = self.emb(x_tok)                      # [B, Ttok, D]

        # Build simple per-step inputs (reuse e as constant "context")
        # We'll pool e to a single context vector to keep it cheap
        ctx = e.mean(dim=1, keepdim=True)       # [B,1,D]

        for _ in range(N):
            # L runs T steps, conditioned on current H
            for _t in range(T):
                l_in = torch.cat([ctx.repeat(1,1,1), zH.transpose(0,1)], dim=-1)  # [B,1,2D]
                _, zL = self.l_rnn(l_in, zL)      # zL: [1,B,D]
                zL = self.norm_l(zL.transpose(0,1)).transpose(0,1)

            # One H update using current L
            h_in = torch.cat([ctx.repeat(1,1,1), zL.transpose(0,1)], dim=-1)      # [B,1,2D]
            _, zH = self.h_rnn(h_in, zH)          # zH: [1,B,D]
            zH = self.norm_h(zH.transpose(0,1)).transpose(0,1)

        h_final = zH.squeeze(0)                  # [B,D]
        s = self.scorer(h_final).squeeze(-1)     # [B]
        q = self.q_head(h_final) if self.use_act else None  # [B,2] (halt/continue)
        return (zL, zH), s, q

    def forward(self, x_tok: torch.LongTensor,
                N: int = 3, T: int = 4,
                z_init: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Inference convenience (no ACT): return scores only.
        """
        B = x_tok.size(0)
        D = self.emb.embedding_dim
        zL = torch.zeros(1, B, D, device=x_tok.device) if z_init is None else z_init[0]
        zH = torch.zeros(1, B, D, device=x_tok.device) if z_init is None else z_init[1]
        (_, _), s, _ = self.forward_once(x_tok, zL, zH, N, T)
        return s

class _HRMDataset(Dataset):
    """
    Loads either pairwise or pointwise labels from CSV.

    Pairwise CSV header example:
        pid_pos,pid_neg
    Pointwise CSV header example:
        pid,label   # label in {0,1}

    During training, paper tokenization is stubbed in _train_neural_hrm.
    Replace with your full tokenization if you have the full paper maps here.
    """
    def __init__(self, csv_path: str):
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = [h.lower() for h in reader.fieldnames or []]
            if "pid_pos" in header and "pid_neg" in header:
                self.mode = "pair"
                for r in reader:
                    rows.append((r["pid_pos"], r["pid_neg"]))
            elif "pid" in header and "label" in header:
                self.mode = "point"
                for r in reader:
                    rows.append((r["pid"], int(r["label"])))
            else:
                self.mode = "none"
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self.mode == "pair":
            return self.rows[idx][0], self.rows[idx][1]
        elif self.mode == "point":
            return self.rows[idx][0], self.rows[idx][1]
        else:
            raise IndexError("Dataset mode is none / empty")

# ---------------
# END OF MODULE
# ---------------