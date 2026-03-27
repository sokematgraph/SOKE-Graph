import sys
import os
import json
from typing import List, Dict, Set, Any
from collections import defaultdict
import numpy as np
from sokegraph.util.logger import LOG

class SemanticSimilarityMatcher:
    def __init__(self, base_ontology_path: str, similarity_threshold: float = 0.75):
        self.base_ontology_path = base_ontology_path
        self.similarity_threshold = similarity_threshold
        self.backend = os.getenv("SOKEGRAPH_SEM_BACKEND", "embedding").strip().lower()
        if sys.platform == "darwin" and not os.getenv("SOKEGRAPH_SEM_BACKEND"):
             self.backend = "lexical" # avoid fault 11 if not explicitly forced
             
        self.ontology_terms = self._load_ontology_terms()
        
        if self.backend == "embedding":
            try:
                from sentence_transformers import SentenceTransformer
                LOG.info("Loading SentenceTransformer for semantic matching (device=cpu)")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                LOG.info(f"Computing embeddings for {len(self.ontology_terms)} ontology terms...")
                self.ontology_embeddings = self.model.encode(self.ontology_terms, convert_to_tensor=True, device="cpu")
            except Exception as e:
                LOG.warning(f"Failed to load standard embedding backend: {e}. Falling back to lexical.")
                self.backend = "lexical"

        if self.backend == "lexical":
            try:
                from rapidfuzz import fuzz
                self.fuzz = fuzz
                LOG.info("Using RapidFuzz backend for lexical matching.")
            except ImportError:
                LOG.warning("Rapidfuzz not installed. Semantic matching will be exact-match only.")
                self.fuzz = None

    def _load_ontology_terms(self) -> List[str]:
        terms = set()
        data = None
        try:
            # Load from file path directly
            with open(self.base_ontology_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if isinstance(data, dict) and "@graph" in data:
                # Parse JSON-LD structure
                for node in data.get("@graph", []):
                    if "skos:prefLabel" in node:
                        lbl = node["skos:prefLabel"]
                        if isinstance(lbl, str): terms.add(lbl.lower())
                        elif isinstance(lbl, list): terms.update(l.lower() for l in lbl)
                    if "skos:altLabel" in node:
                        alt = node["skos:altLabel"]
                        if isinstance(alt, str): terms.add(alt.lower())
                        elif isinstance(alt, list): terms.update(a.lower() for a in alt)
            else:
                # Parse legacy dictionary format
                def parse_dict(d):
                    for k, v in d.items():
                        if isinstance(v, list):
                            for term in v:
                                if isinstance(term, str): terms.add(term.lower())
                        elif isinstance(v, dict):
                            parse_dict(v)
                parse_dict(data)
                
        except Exception as e:
            LOG.error(f"Failed to load ontology terms for semantic matcher from {self.base_ontology_path}: {e}")
            
        return list(terms)

    def get_synonyms(self, query_keywords: List[str]) -> Dict[str, List[str]]:
        """Find matching ontology terms for each query keyword."""
        results = defaultdict(list)
        
        if not self.ontology_terms:
            return results

        if self.backend == "embedding" and hasattr(self, "model"):
            import torch
            query_embeddings = self.model.encode(query_keywords, convert_to_tensor=True, device="cpu")
            from sentence_transformers.util import cos_sim
            cosine_scores = cos_sim(query_embeddings, self.ontology_embeddings)
            
            for i, query in enumerate(query_keywords):
                scores = cosine_scores[i]
                mask = scores >= self.similarity_threshold
                indices = torch.nonzero(mask).squeeze(-1)
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                    
                for idx in indices:
                    results[query].append(self.ontology_terms[idx])

        elif self.backend == "lexical" and hasattr(self, "fuzz") and self.fuzz:
            for query in query_keywords:
                for term in self.ontology_terms:
                    score = self.fuzz.token_sort_ratio(query.lower(), term.lower()) / 100.0
                    if score >= self.similarity_threshold:
                        results[query].append(term)
        else:
            for query in query_keywords:
                ql = query.lower()
                for term in self.ontology_terms:
                    if ql in term or term in ql:
                        results[query].append(term)
                        
        LOG.info(f"DEBUG: Semantic matching found synonyms: {dict(results)}")
        return dict(results)
