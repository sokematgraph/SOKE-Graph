import json
from collections import defaultdict
from pathlib import Path
import networkx as nx
from sokegraph.knowledge_graph import KnowledgeGraph
from pyvis.network import Network
import streamlit as st
import tempfile
import uuid
from typing import Optional, Set
from pyvis.network import Network

CATEGORY_COLOR = {
    "Layer": "#FFD700",
    "Category": "#FF7F50",
    "Keyword": "#87CEFA",
    "Paper": "#90EE90",
    "MetaData": "#D3D3D3",
    "_default": "#CCCCCC"
}

class NetworkXKnowledgeGraph(KnowledgeGraph):
    """
    Concrete implementation of KnowledgeGraph that builds the same
    Layer→Category→Keyword→MetaData/Paper structure, but in a
    NetworkX MultiDiGraph instead of Neo4j.

    Node "labels" from Neo4j are stored in the `kind` attribute.
    Every edge gets a `rel` attribute carrying the relationship type.
    """

    def __init__(self, ontology_path: str):
        super().__init__(ontology_path)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def build_graph(self) -> nx.MultiDiGraph:
        unique_layers: set[str] = set()
        unique_categories: set[str] = set()
        unique_keywords: dict[tuple[str, str], set[str]] = defaultdict(set)
        unique_metadata: set[str] = set()
        paper_keywords: dict[str, set[str]] = defaultdict(set)

        for layer, categories in self.ontology_extractions.items():
            self._add_node_once(
                key=layer,
                kind="Layer",
                attrs={"name": layer},
                registry=unique_layers,
            )

            for cat, items in categories.items():
                cat_key = f"{layer}|{cat}"
                self._add_node_once(
                    key=cat_key,
                    kind="Category",
                    attrs={"name": cat, "layer": layer},
                    registry=unique_categories,
                )
                self._add_edge_once(layer, cat_key, "HAS_CATEGORY")

                for item in items:
                    paper_id = item.get("paper_id", "unknown")

                    for kw in item["keywords"]:
                        kw_key = f"{layer}|{cat}|{kw}"
                        if kw_key not in unique_keywords[(layer, cat)]:
                            unique_keywords[(layer, cat)].add(kw_key)
                            self._add_node_once(
                                key=kw_key,
                                kind="Keyword",
                                attrs={"name": kw, "layer": layer, "category": cat},
                            )
                            self._add_edge_once(cat_key, kw_key, "HAS_KEYWORD")

                        paper_keywords[paper_id].add(kw_key)

                        for meta in item.get("parsed_meta", []):
                            print(f"meta : {meta}")    
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
                                }#,
                                #registry=unique_metadata,
                            )
                            self._add_edge_once(kw_key, md_key, "HAS_METADATA")

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

    def _add_node_once(self, key: str, kind: str, attrs: dict, registry: Optional[Set[str]] = None) -> None:
        if registry is not None and key in registry:
            print(key in registry)
            print("not add")
            return
        self.graph.add_node(key, kind=kind, **attrs)
        if registry is not None:
            registry.add(key)

    def _add_edge_once(self, src: str, tgt: str, rel: str) -> None:
        for _, target, data in self.graph.out_edges(src, data=True):
            if data.get("rel") == rel and target == tgt:
                return
        self.graph.add_edge(src, tgt, rel=rel)

    def show_graph(self):
        net = Network(height="600px", width="100%", directed=True)
        net.barnes_hut()

        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get("label", str(node))
            net.add_node(
                n_id=node,
                label=label,
                title=json.dumps(attrs, indent=2),
                color="#1f78b4"
            )

        for source, target, attrs in self.graph.edges(data=True):
            label = attrs.get("label", "")
            net.add_edge(
                source,
                target,
                label=label,
                title=json.dumps(attrs, indent=2),
                color="#aaaaaa"
            )

        html_content = net.generate_html()
        tmp_path = Path(tempfile.gettempdir()) / f"graph_{uuid.uuid4().hex}.html"
        tmp_path.write_text(html_content, encoding="utf-8")
        st.components.v1.html(html_content, height=650, scrolling=True)

    def get_attr_keys(self):
        keys = set()
        for _, d in self.graph.nodes(data=True):
            keys.update(d.keys())
        return sorted(keys)
    
    def get_attr_values(self, kind: str) -> list:
        print(kind)
        return [
            data.get("name", str(n))
            for n, data in self.graph.nodes(data=True)
            if data.get("kind") == kind
        ]

    def subgraph_for_attr(self, key, value):
        nodes = [n for n, d in self.graph.nodes(data=True) if d.get(key) == value]
        return self.graph.subgraph(nodes).copy()

    def neighbour_subgraph(self, node_id):
        return self.graph.subgraph([node_id, *self.graph.neighbors(node_id)]).copy()

    def get_node_labels(self):
        kinds = set()
        for _, data in self.graph.nodes(data=True):
            kind = data.get("kind")
            if kind:
                kinds.add(kind)
        return sorted(kinds)
    
    



    def generate_nodes_html(self, node_type: str) -> str:
        """
        Return PyVis HTML containing *only nodes* (no edges).

        Parameters
        ----------
        node_type : str
            One of "All", "Layer", "Category", "Keyword", "Paper", "MetaData".
        """
        net = Network(height="650px", width="100%", directed=False)
        net.barnes_hut()  # layout

        for node_id, data in self.graph.nodes(data=True):
            label = data.get("kind", "Unknown")

            if node_type == "All" or label == node_type:
                name = data.get("name", str(node_id))
                color = CATEGORY_COLOR.get(label, CATEGORY_COLOR["_default"])
                net.add_node(node_id, label=name, color=color, title=label)

        return net.generate_html()
    
    def fetch_related(self, node_name: str):
        """
        Fetch neighbors of a given node along with relationship type and direction.

        Parameters
        ----------
        node_name : str
            The 'name' attribute of the target node.

        Returns
        -------
        List[Dict]
            Each dict contains:
            - neighbour: name of the neighbor node
            - rel_type: value of the 'type' attribute on the edge
            - direction: 'out' if edge goes from node_name to neighbor, 'in' otherwise
        """
        related = []

        # Find the actual node ID that has the given name
        target_id = None
        for nid, data in self.graph.nodes(data=True):
            if data.get("name") == node_name:
                target_id = nid
                break

        if target_id is None:
            return []

        # Outgoing edges
        for _, target, edge_data in self.graph.out_edges(target_id, data=True):
            neighbor_name = self.graph.nodes[target].get("name", str(target))
            rel_type = edge_data.get("rel", "related_to")
            related.append({
                "neighbour": neighbor_name,
                "rel_type": rel_type,
                "direction": "out"
            })

        # Incoming edges
        for source, _, edge_data in self.graph.in_edges(target_id, data=True):
            neighbor_name = self.graph.nodes[source].get("name", str(source))
            rel_type = edge_data.get("rel", "related_to")
            related.append({
                "neighbour": neighbor_name,
                "rel_type": rel_type,
                "direction": "in"
            })

        # Optional: sort by relationship type and neighbor name
        related.sort(key=lambda x: (x["rel_type"], x["neighbour"]))
        return related


