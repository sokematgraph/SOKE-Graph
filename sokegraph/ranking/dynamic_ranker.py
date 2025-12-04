"""
dynamic_ranker.py

Implements :class:`DynamicRanker`, an advanced paper ranking method using:

1. Dynamic adaptive thresholds based on paper-keyword pair distributions
2. IDF-weighted scoring for keyword importance
3. Category-based relevance analysis
4. Overlap detection across ranking approaches (static vs dynamic vs HRM)

This ranker automatically determines quality thresholds from the data
rather than using fixed cutoffs, making it more robust across different
paper sets and domains.

Classes:
- DynamicRanker: Dynamic adaptive paper ranking with IDF weighting
"""

import os
from sokegraph.ranking.base_ranking_method import BaseRankingMethod
from sokegraph.agents.ai_agent import AIAgent

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


class DynamicRanker(BaseRankingMethod):
    
    def __init__(self,
        ai_tool: AIAgent,
        papers_path,
        ontology_path,
        keyword_path,
        output_dir: str):
        super().__init__(ai_tool, papers_path, ontology_path, keyword_path, output_dir)
        
        

    def rank(self):
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
        dynamic_ranked, low_sorted, pair_file_path = self.find_common_papers(
            category_to_papers,
            paper_keyword_map,
            title_map,
            kw_lookup,
            abstract_map,
        )

        dynamic_outputs_csv,  dynamic_outputs_all = self.rank_shared_papers_by_pairs_and_mentions(
            dynamic_ranked, low_sorted, pair_file_path, self.output_dir, label="dynamic"
        )
        
        return dynamic_outputs_csv, dynamic_outputs_all

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
        trimmed = [x for x in arr if (x >= low and x <= high)]
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
        dynamic_scores = defaultdict(float)
        paper_keyword_frequencies = defaultdict(lambda: defaultdict(int))
        filtered_out = {}

        for (layer, cat), hits in category_to_papers.items():
            if (layer, cat) not in [(l, c) for l, c, _ in categories]:
                continue

            per_cat_hits[(layer, cat)] = hits
            for pid, _count in hits.items():
                kw_set = paper_keyword_map[(layer, cat)][pid]
                dyn_inc = 0.0
                for kw in kw_set:
                    if kw in idf:
                        print(f"inside if 1, kw : {kw}")
                        dyn_inc += idf[kw]
                        paper_keyword_frequencies[pid][kw] += 1
                dynamic_scores[pid] += dyn_inc
                print(f"kw_set : {kw_set}, dyn_inc : {dyn_inc}")
                print(f"pid : {pid}, dynamic_scores[pid] : {dynamic_scores[pid]}")
        

        # ---- Dynamic survivors with τ + opposites ----
        tau_dynamic = self._dynamic_threshold(list(dynamic_scores.values()))
        LOG.info(f"Dynamic threshold (τ) = {tau_dynamic:.3f}")

        dynamic_ranked = []
        for pid, score in dynamic_scores.items():
            if score <= 0:
                continue
            if not self.is_dominated_by_opposites(
                pid, title_map, abstract_map, query_keywords, expanded_keywords, opposites, filtered_out
            ):
                if score >= tau_dynamic:
                    dynamic_ranked.append((pid, float(score)))

        # Persist τ
        try:
            pd.DataFrame([{"tau_dynamic": tau_dynamic}]).to_csv(
                f"{self.output_dir}/dynamic_threshold.csv", index=False
            )
        except Exception:
            pass

        ranked_paper_ids = {pid for pid, _ in set(dynamic_ranked)}

        # ---- Export filtered-out breakdown (optional) ----
        if filtered_out:
            summary_df = BaseRankingMethod.summarize_filtered_papers_with_opposites(
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
            return dynamic_ranked, low_sorted, df_pairs_file_path

        ranked_by_pair_overlap, total_possible_pairs = BaseRankingMethod.rank_by_pair_overlap_filtered(
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


        return dynamic_ranked, low_sorted, df_pairs_file_path

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

    def is_dominated_by_opposites(
        self,
        paper_id: str,
        title_map: Dict[str, str],
        abstract_map: Dict[str, str],
        query_keywords: List[str],
        expanded_keywords: Dict[str, List[str]],
        opposites: Dict[str, List[str]],
        filtered_out: Dict[str, Dict]
    ) -> bool:
        title_text = (title_map.get(paper_id, "") or "").lower()
        full_abstract = (str(abstract_map.get(paper_id, "") or "")).lower()
        abstract_text = re.split(r'(references|refs?\.?:)', full_abstract)[0]

        title_flags: Dict[str, str] = {}
        title_over_threshold = 0

        for qk in query_keywords:
            relevant_terms = expanded_keywords.get(qk, [qk])
            opp_terms = opposites.get(qk, [])
            title_rel = sum(
                len(re.findall(rf'\b{re.escape(term)}\b', title_text))
                for term in relevant_terms
            )
            title_opp = sum(
                len(re.findall(rf'\b{re.escape(opp)}\b', title_text))
                for opp in opp_terms
            )
            if title_rel == 0 and title_opp == 0:
                title_flags[qk] = "missing"
            else:
                ratio = float('inf') if title_rel == 0 else title_opp / title_rel
                if ratio > 1:
                    title_flags[qk] = "fail"
                    title_over_threshold += 1
                elif ratio in (0, 1):
                    title_flags[qk] = "pass"
                else:
                    title_flags[qk] = "other"

        if "fail" in title_flags.values():
            filtered_out[paper_id] = {
                "reason": "title_fail",
                "title_flags": title_flags,
                "title_exceeded": title_over_threshold,
                "threshold": 1.0,
            }
            return True
        
        if "pass" in title_flags.values() and "missing" in title_flags.values():
            abstract_ratios = []
            for qk in query_keywords:
                relevant_terms = expanded_keywords.get(qk, [qk])
                opp_terms = opposites.get(qk, [])
                abs_rel = sum(
                    len(re.findall(rf'\b{re.escape(term)}\b', abstract_text))
                    for term in relevant_terms
                )
                abs_opp = sum(
                    len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text))
                    for opp in opp_terms
                )
                abstract_ratios.append(ratio)

            if all(r <= 1.5 for r in abstract_ratios):
                return False
            else:
                filtered_out[paper_id] = {
                    "reason": "moderate_abstract_check",
                    "abstract_ratios": abstract_ratios,
                    "title_exceeded": 0,
                    "threshold": 1.5,
                }
                return False

        if all(flag == "pass" for flag in title_flags.values()):
            return False

        return False

    
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
            on="paper_id",     
            how="left"
        )


        output_paths_all = {}
        output_csv_paths = {}
        for level in ["high", "low", "unknown"]:
            subset = df_merged[df_merged["Relevance Level"] == level]
            if not subset.empty:
                basename = f"shared_ranked_by_pairs_then_mentions_{label}_{level}"
                paths = BaseRankingMethod.save_papers(subset, self.output_dir, basename)
                output_paths_all[label+"_"+level] = paths
                output_csv_paths[label+"_"+level] = paths["csv"]  # only CSV
                LOG.info(f"✅ Saved CSV: {paths['csv']}")

        return output_csv_paths, output_paths_all

    

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