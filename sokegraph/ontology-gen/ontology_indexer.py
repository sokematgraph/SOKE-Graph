#!/usr/bin/env python3
"""Build a persistent FAISS semantic index from ontology JSON-LD."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_index(ontology_jsonld: str, output_dir: str) -> tuple[str, str]:
    ontology_path = Path(ontology_jsonld)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ontology_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    texts = []
    mapping = {}

    for idx, node in enumerate(payload.get("@graph", [])):
        iri = node.get("@id")
        pref = (node.get("skos:prefLabel") or "").strip()
        alt = node.get("skos:altLabel") or []
        if isinstance(alt, str):
            alt = [alt]
        alt = [str(a).strip() for a in alt if str(a).strip()]

        if not iri or not pref:
            continue

        text = " | ".join([pref, *alt])
        mapping[len(texts)] = {
            "@id": iri,
            "prefLabel": pref,
            "altLabels": alt,
            "@type": node.get("@type", ""),
        }
        texts.append(text)

    if not texts:
        raise ValueError("No valid concepts found in @graph.")

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = out_dir / "ontology.index"
    mapping_path = out_dir / "ontology_mapping.pkl"

    faiss.write_index(index, str(index_path))
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f)

    return str(index_path), str(mapping_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ontology FAISS index from ontology.jsonld")
    parser.add_argument("--ontology", required=True, help="Path to ontology.jsonld")
    parser.add_argument("--out", default="external/output/semantic_index", help="Output directory")
    args = parser.parse_args()

    idx_path, map_path = build_index(args.ontology, args.out)
    print(f"Saved index: {idx_path}")
    print(f"Saved mapping: {map_path}")


if __name__ == "__main__":
    main()
