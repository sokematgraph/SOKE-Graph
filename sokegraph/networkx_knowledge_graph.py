import json
from collections import defaultdict
from pathlib import Path
import networkx as nx
from sokegraph.knowledge_graph import KnowledgeGraph
import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile
import uuid


class NetworkXKnowledgeGraph(KnowledgeGraph):
    """
    Concrete implementation of KnowledgeGraph that builds the same
    Layer → Category → Keyword → MetaData / Paper structure, but in a
    NetworkX MultiDiGraph instead of Neo4j.

    Node “labels” from Neo4j are stored in the `kind` attribute.
    Every edge gets a `rel` attribute carrying the relationship type.
    """

    def __init__(self, ontology_path: str):
        """
        Parameters
        ----------
        ontology_path
            Path to the ontology‑extractions JSON file.
            The parent `KnowledgeGraph` takes care of loading it and
            populating `self.ontology_extractions`.
        """
        super().__init__(ontology_path)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build_graph(self) -> nx.MultiDiGraph:
        """
        Build and return the directed multigraph.

        The method tracks created nodes with Python sets to avoid
        accidental duplication—similar to `MERGE` in Neo4j.
        """
        unique_layers: set[str] = set()
        unique_categories: set[str] = set()                # layer|cat
        unique_keywords: dict[tuple[str, str], set[str]] = defaultdict(set)
        unique_metadata: set[str] = set()                  # kw_key|unit|value
        paper_keywords: dict[str, set[str]] = defaultdict(set)

        for layer, categories in self.ontology_extractions.items():
            # ----------------------------------------------------------
            # LAYER
            # ----------------------------------------------------------
            self._add_node_once(
                key=layer,
                kind="Layer",
                attrs={"name": layer},
                registry=unique_layers,
            )

            for cat, items in categories.items():
                cat_key = f"{layer}|{cat}"

                # ------------------------------------------------------
                # CATEGORY
                # ------------------------------------------------------
                self._add_node_once(
                    key=cat_key,
                    kind="Category",
                    attrs={"name": cat, "layer": layer},
                    registry=unique_categories,
                )
                self._add_edge_once(layer, cat_key, "HAS_CATEGORY")

                for item in items:
                    paper_id = item.get("paper_id", "unknown")

                    # --------------------------------------------------
                    # KEYWORDS
                    # --------------------------------------------------
                    for kw in item["keywords"]:
                        kw_key = f"{layer}|{cat}|{kw}"
                        if kw_key in unique_keywords[(layer, cat)]:
                            pass  # already handled
                        else:
                            unique_keywords[(layer, cat)].add(kw_key)
                            self._add_node_once(
                                key=kw_key,
                                kind="Keyword",
                                attrs={"name": kw, "layer": layer, "category": cat},
                            )
                            self._add_edge_once(cat_key, kw_key, "HAS_KEYWORD")

                        # collect for Paper‑>Keyword step later
                        paper_keywords[paper_id].add(kw_key)

                        # --------------------------------------------------
                        # METADATA
                        # --------------------------------------------------
                        for meta in item.get("parsed_meta", []):
                            md_key = f"{kw_key}|{meta['unit']}|{meta['value']}"
                            if md_key in unique_metadata:
                                continue
                            unique_metadata.add(md_key)

                            self._add_node_once(
                                key=md_key,
                                kind="MetaData",
                                attrs={
                                    "name": meta["unit"],
                                    "unit": meta["unit"],
                                    "value": meta["value"],
                                    "category": cat,
                                },
                                registry=unique_metadata,  # no effect, keeps signature tidy
                            )
                            self._add_edge_once(kw_key, md_key, "HAS_METADATA")

        # --------------------------------------------------------------
        # PAPERS & MENTIONS
        # --------------------------------------------------------------
        for paper_id, kw_keys in paper_keywords.items():
            self._add_node_once(
                key=paper_id,
                kind="Paper",
                attrs={"name": paper_id},
            )
            for kw_key in kw_keys:
                self._add_edge_once(kw_key, paper_id, "MENTIONS")

        print("🎉 NetworkX knowledge graph construction complete.")
        return self.graph

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------



    from typing import Optional, Set

    
    def _add_node_once(
        self,
        key: str,
        kind: str,
        attrs: dict,
        registry: Optional[Set[str]] = None
    ) -> None:
        """
        Add a node only if it hasn’t been seen before.
        """
        if registry is not None and key in registry:
            return
        self.graph.add_node(key, kind=kind, **attrs)
        if registry is not None:
            registry.add(key)

    def _add_edge_once(self, src: str, tgt: str, rel: str) -> None:
        """
        Add an edge with a `rel` attribute, but skip duplicate
        (src, tgt, rel) triples.
        """
        # NetworkX MultiDiGraph stores edges with keys; iterate to see if identical already exists
        for _, _, data in self.graph.out_edges(src, data=True):
            if data.get("rel") == rel and _ == tgt:
                return
        self.graph.add_edge(src, tgt, rel=rel)

    def show_graph(self):
        net = Network(height="600px", width="100%", directed=True)
        net.barnes_hut()

        # --- Add nodes with labels ---
        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get("label", str(node))  # fallback to node ID
            net.add_node(
                n_id=node,
                label=label,
                title=json.dumps(attrs, indent=2),  # shows full metadata on hover
                color="#1f78b4"
            )

        # --- Add edges with optional labels ---
        for source, target, attrs in self.graph.edges(data=True):
            label = attrs.get("label", "")
            net.add_edge(
                source,
                target,
                label=label,
                title=json.dumps(attrs, indent=2),  # hover shows full edge data
                color="#aaaaaa"
            )

        # --- Generate HTML content ---
        html_content = net.generate_html()

        # --- Optional: Write to tmp file (but not required anymore) ---
        tmp_path = Path(tempfile.gettempdir()) / f"graph_{uuid.uuid4().hex}.html"
        tmp_path.write_text(html_content, encoding="utf-8")

        # --- Show in Streamlit ---
        st.components.v1.html(html_content, height=650, scrolling=True)

        
    def get_attr_keys(self):
        keys = set()
        for _, d in self._nx.nodes(data=True):
            keys.update(d.keys())
        return sorted(keys)

    def get_attr_values(self, key):
        return sorted({d.get(key) for _, d in self._nx.nodes(data=True) if key in d})

    def subgraph_for_attr(self, key, value):
        nodes = [n for n, d in self._nx.nodes(data=True) if d.get(key) == value]
        return self._nx.subgraph(nodes).copy()

    def neighbour_subgraph(self, node_id):
        return self._nx.subgraph([node_id, *self._nx.neighbors(node_id)]).copy()