# paper_ranker/base_ranking_method.py
from abc import ABC, abstractmethod
from sokegraph.agents.ai_agent import AIAgent
import json
import os
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

class BaseRankingMethod(ABC):

    def __init__(self,
        ai_tool: AIAgent,
        papers_path,
        ontology_path,
        keyword_path,
        output_dir: str):
        self.ai_tool = ai_tool
        self.papers = load_papers(papers_path)
        self.ontology_path = ontology_path
        self.ontology = None  # set by _load_ontology
        self._load_ontology()
        self.keyword_query = load_keyword(keyword_path)
        self.output_dir = output_dir
    

    def _load_ontology(self):
        """Load ontology JSON from self.ontology_path into self.ontology."""
        try:
            with open(self.ontology_path, "r", encoding="utf-8") as f:
                self.ontology = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load ontology from '{self.ontology_path}': {e}")
        
    @abstractmethod
    def rank(self):
        pass
    

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

        if "paper_id" in df.columns:
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_node(row["paper_id"], **row.to_dict())
            graphml_path = os.path.join(output_dir, f"{basename}.graphml")
            nx.write_graphml(G, graphml_path)
            paths["graphml"] = graphml_path

        return paths
    

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
            abstract_text = (abstract_map.get(pid, "") or "").lower()
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
    


    def _get_title_map(self):
        """
        Map: safe title or paper_id -> title text
        """
        title_map = {}
        for _, row in self.papers.iterrows():
            safe_id = str(row.get("paper_id", ""))
            # Store the abstract (or empty string if missing) as the value
            title_map[safe_id] = row.get("title", "")
        return title_map


    def _get_abstract_map(self):
        """
         Map: safe title or paper_id -> abstract text
        """
        abstract_map = {}
        for _, row in self.papers.iterrows():
            # Normalize title or use paper_id, just like in _get_title_map
            safe_id = str(row.get("paper_id", ""))
            # Assign abstract text (or an empty string if not available)
            abstract_map[safe_id] = row.get("abstract", "")
        return abstract_map
