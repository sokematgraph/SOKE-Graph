from __future__ import annotations

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

LOG = logging.getLogger("sokegraph")


TYPE_TO_LAYER = {
    "Device": "Device",
    "Environment": "Environment",
    "Process": "Process",
    "Reaction": "Reaction",
    "Element": "Elemental Composition",
    "Material": "Material",
    "PerformanceMetric": "Performance & Stability",
    "Application": "Application",
}


@dataclass
class SemanticMatch:
    iri: str
    pref_label: str
    concept_type: str
    layer: str
    score: float
    sentence: str


class OntologySemanticIndexer:
    def __init__(
        self,
        ontology_jsonld_path: str,
        index_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        backend: str = "embedding",
    ) -> None:
        self.ontology_jsonld_path = Path(ontology_jsonld_path)
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.device = device
        self.backend = backend

        # Prevent tokenizer thread oversubscription / instability in some environments.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "ontology_semantic.index"
        self.mapping_path = self.index_dir / "ontology_semantic_mapping.pkl"

        self.model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.mapping: Dict[int, dict] = {}

    def _ensure_model(self) -> SentenceTransformer:
        if self.model is None:
            LOG.info("Loading semantic encoder '%s' on device=%s", self.model_name, self.device)
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def _load_jsonld(self) -> dict:
        with open(self.ontology_jsonld_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _concept_rows(self) -> List[Tuple[str, dict]]:
        payload = self._load_jsonld()
        rows: List[Tuple[str, dict]] = []

        for node in payload.get("@graph", []):
            iri = node.get("@id")
            pref = (node.get("skos:prefLabel") or "").strip()
            if not iri or not pref:
                continue

            alt = node.get("skos:altLabel") or []
            if isinstance(alt, str):
                alt = [alt]
            alt = [str(a).strip() for a in alt if str(a).strip()]

            text = " | ".join([pref, *alt])
            concept_type = str(node.get("@type", "")).strip()
            layer = TYPE_TO_LAYER.get(concept_type, concept_type or "Unknown")

            rows.append(
                (
                    text,
                    {
                        "iri": iri,
                        "pref_label": pref,
                        "alt_labels": alt,
                        "concept_type": concept_type,
                        "layer": layer,
                    },
                )
            )
        return rows

    def build_index(self, force_rebuild: bool = False) -> None:
        if self.index_path.exists() and self.mapping_path.exists() and not force_rebuild:
            self.load_index()
            return

        rows = self._concept_rows()
        if not rows:
            raise ValueError("No ontology concepts found in JSON-LD @graph.")

        LOG.info("Building semantic ontology index from %s (%d concepts)", self.ontology_jsonld_path, len(rows))

        model = self._ensure_model()
        texts = [row[0] for row in rows]
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        vectors = np.asarray(vectors, dtype="float32")

        dim = int(vectors.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        mapping = {i: meta for i, (_, meta) in enumerate(rows)}

        faiss.write_index(index, str(self.index_path))
        with open(self.mapping_path, "wb") as f:
            pickle.dump(mapping, f)

        self.index = index
        self.mapping = mapping

    def load_index(self) -> None:
        if not self.index_path.exists() or not self.mapping_path.exists():
            self.build_index(force_rebuild=True)
            return

        self.index = faiss.read_index(str(self.index_path))
        with open(self.mapping_path, "rb") as f:
            self.mapping = pickle.load(f)
        LOG.info("Loaded semantic ontology index: %s", self.index_path)

    def ensure_ready(self, force_rebuild: bool = False) -> None:
        if self.backend == "lexical":
            if not self.mapping:
                rows = self._concept_rows()
                self.mapping = {i: meta for i, (_, meta) in enumerate(rows)}
            return

        if force_rebuild:
            self.build_index(force_rebuild=True)
            return
        if self.index is None or not self.mapping:
            self.load_index()

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text or "")
        return [p.strip() for p in parts if p and p.strip()]

    def search_text(
        self,
        text: str,
        k: int = 1,
        similarity_threshold: float = 0.62,
    ) -> List[SemanticMatch]:
        if self.backend == "lexical":
            return self._search_text_lexical(text=text, k=k, similarity_threshold=similarity_threshold)

        self.ensure_ready()
        model = self._ensure_model()

        sentences = self.split_sentences(text)
        if not sentences:
            return []

        q = model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q = np.asarray(q, dtype="float32")

        scores, ids = self.index.search(q, k)  # type: ignore[union-attr]
        matches: List[SemanticMatch] = []
        seen = set()

        for i, sentence in enumerate(sentences):
            for j in range(k):
                idx = int(ids[i][j])
                score = float(scores[i][j])
                if idx < 0 or score < similarity_threshold:
                    continue

                meta = self.mapping.get(idx)
                if not meta:
                    continue

                key = (meta["iri"], sentence)
                if key in seen:
                    continue
                seen.add(key)

                matches.append(
                    SemanticMatch(
                        iri=meta["iri"],
                        pref_label=meta["pref_label"],
                        concept_type=meta["concept_type"],
                        layer=meta["layer"],
                        score=score,
                        sentence=sentence,
                    )
                )
        return matches

    def _search_text_lexical(
        self,
        text: str,
        k: int = 1,
        similarity_threshold: float = 0.62,
    ) -> List[SemanticMatch]:
        self.ensure_ready()
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        # Map threshold from [0,1] to RapidFuzz [0,100]
        min_score = max(0.0, min(100.0, similarity_threshold * 100.0))
        matches: List[SemanticMatch] = []

        for sentence in sentences:
            scored = []
            s = sentence.strip().lower()
            if not s:
                continue

            for idx, meta in self.mapping.items():
                pool = [meta.get("pref_label", "")] + list(meta.get("alt_labels", []) or [])
                best = 0.0
                for candidate in pool:
                    c = str(candidate).strip().lower()
                    if not c:
                        continue
                    best = max(best, float(fuzz.token_set_ratio(s, c)))
                if best >= min_score:
                    scored.append((best, idx))

            scored.sort(key=lambda x: x[0], reverse=True)
            for best, idx in scored[: max(1, k)]:
                meta = self.mapping[idx]
                matches.append(
                    SemanticMatch(
                        iri=meta["iri"],
                        pref_label=meta["pref_label"],
                        concept_type=meta["concept_type"],
                        layer=meta["layer"],
                        score=best / 100.0,
                        sentence=sentence,
                    )
                )

        return matches

    def map_text_to_iris(
        self,
        text: str,
        k: int = 1,
        similarity_threshold: float = 0.62,
    ) -> List[str]:
        matches = self.search_text(text=text, k=k, similarity_threshold=similarity_threshold)
        uniq = []
        seen = set()
        for m in matches:
            if m.iri not in seen:
                seen.add(m.iri)
                uniq.append(m.iri)
        return uniq


def build_semantic_extractions(
    papers: pd.DataFrame | List[Dict[str, Any]],
    indexer: OntologySemanticIndexer,
    similarity_threshold: float = 0.62,
) -> Dict[str, Dict[str, List[dict]]]:
    """Create ontology-extraction payload expected by SOKEGraph from paper abstracts."""
    if isinstance(papers, list):
        papers_df = pd.DataFrame(papers)
    else:
        papers_df = papers.copy()

    results: Dict[str, Dict[str, List[dict]]] = {}

    for _, row in papers_df.iterrows():
        paper_id = str(row.get("paper_id", "") or "")
        title = str(row.get("title", "") or paper_id)
        abstract = str(row.get("abstract", "") or "")
        if not abstract.strip():
            continue

        matches = indexer.search_text(
            text=abstract,
            k=1,
            similarity_threshold=similarity_threshold,
        )

        for m in matches:
            layer = m.layer or "Unknown"
            category = m.pref_label or "Unknown"

            if layer not in results:
                results[layer] = {}
            if category not in results[layer]:
                results[layer][category] = []

            results[layer][category].append(
                {
                    "keywords": [m.pref_label],
                    "keyword_ids": [m.iri],
                    "concept_id": m.iri,
                    "concept_label": m.pref_label,
                    "meta_data": m.sentence,
                    "paper_id": paper_id,
                    "paper_title": title,
                    "match_score": m.score,
                    "parsed_meta": [],
                }
            )

    return results
