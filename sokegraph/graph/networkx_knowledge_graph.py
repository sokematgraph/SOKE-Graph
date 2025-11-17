"""
NetworkXKnowledgeGraph
----------------------
A concrete KnowledgeGraph implementation using NetworkX MultiDiGraph.

Features
- Builds Layer → Category → Keyword → (MetaData / Paper) structure
- Colors by node 'kind' using self.CATEGORY_COLOR
- Streamlit + Matplotlib visualizations
- Interactive Jupyter subgraph explorer with attributes table
"""

from __future__ import annotations


import json
import uuid
import tempfile
import uuid
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Set, Dict, List

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Optional deps for interactive/Jupyter usage
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import pandas as pd
except Exception:  # pragma: no cover
    widgets = None
    display = None
    clear_output = None
    pd = None

# Optional Streamlit visualization
try:
    import streamlit as st
    from pyvis.network import Network
except Exception:  # pragma: no cover
    st = None
    Network = None


# Import your abstract base (keep your original path)
from sokegraph.graph.knowledge_graph import KnowledgeGraph



class NetworkXKnowledgeGraph(KnowledgeGraph):
    """
    Concrete implementation of KnowledgeGraph that builds the same
    Layer→Category→Keyword→MetaData/Paper structure, but in a
    NetworkX MultiDiGraph.

    Node "labels" from Neo4j are stored in the `kind` attribute.
    Every edge gets a `rel` attribute carrying the relationship type.
    """

    def __init__(self, ontology_path: str, papers_path:str):
        super().__init__(ontology_path, papers_path)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ---------- Build ---------------------------------------------------------

    def build_graph(self) -> nx.MultiDiGraph:
        unique_layers: set[str] = set()
        unique_categories: set[str] = set()
        unique_keywords: dict[tuple[str, str], set[str]] = defaultdict(set)
        unique_metadata: set[str] = set()
        paper_keywords: dict[str, set[str]] = defaultdict(set)

        # NEW: remember titles for each paper_id
        paper_titles: dict[str, str] = {}

        for layer, categories in self.ontology_extractions.items():
            print(f"Processing layer: {layer} with categories: {categories}")
            self._add_node_once(
                key=layer, kind="Layer", attrs={"name": layer}, registry=unique_layers
            )

            for cat, items in categories.items():
                cat_key = f"{layer}|{cat}"
                self._add_node_once(
                    key=cat_key, kind="Category",
                    attrs={"name": cat, "layer": layer}, registry=unique_categories
                )
                self._add_edge_once(layer, cat_key, "HAS_CATEGORY")

                for item in items:
                    # --- pull both id and human-readable title ---
                    paper_id = str(item.get("paper_id"))
                    paper_title = self.papers.loc[self.papers["paper_id"] == paper_id, "title"].values[0]
                    print(f"item: {item}")
                    print(f"Processing paper_id: {paper_id}, title: {paper_title}")
                    paper_titles[paper_id] = paper_title  # remember for later
                    print(f"paper_titles : {paper_titles}")

                    for kw in item["keywords"]:
                        kw_key = f"{layer}|{cat}|{kw}"
                        if kw_key not in unique_keywords[(layer, cat)]:
                            unique_keywords[(layer, cat)].add(kw_key)
                            self._add_node_once(
                                key=kw_key, kind="Keyword",
                                attrs={"name": kw, "layer": layer, "category": cat},
                            )
                            self._add_edge_once(cat_key, kw_key, "HAS_KEYWORD")

                        paper_keywords[paper_id].add(kw_key)

                        for meta in item.get("parsed_meta", []):
                            md_key = f"{kw_key}|{meta['unit']}|{meta['value']}"
                            if md_key in unique_metadata:
                                continue
                            unique_metadata.add(md_key)

                            self._add_node_once(
                                key=md_key, kind="MetaData",
                                attrs={
                                    "name": meta["unit"],
                                    "unit": meta["unit"],
                                    "value": meta["value"],
                                    "category": cat,
                                },
                            )
                            self._add_edge_once(kw_key, md_key, "HAS_METADATA")

        # --- create paper nodes using the TITLE as display name ---
        for paper_id, kw_keys in paper_keywords.items():
            display_name = paper_titles.get(paper_id, paper_id)

            self._add_node_once(
                key=paper_id,            # keep id as the unique node key
                kind="Paper",
                attrs={
                    "name": display_name,  # <-- shown in UI / used by your name-based lookups
                    "paper_id": paper_id,  # keep id as an attribute too
                    "title": display_name, # optional alias
                },
            )
            for kw_key in kw_keys:
                self._add_edge_once(kw_key, paper_id, "MENTIONS")

        print("🎉 NetworkX knowledge graph construction complete.")
        return self.graph


    def _add_node_once(self, key: str, kind: str, attrs: dict,
                       registry: Optional[Set[str]] = None) -> None:
        if registry is not None and key in registry:
            return
        self.graph.add_node(key, kind=kind, **attrs)
        if registry is not None:
            registry.add(key)

    def _add_edge_once(self, src: str, tgt: str, rel: str) -> None:
        # avoid duplicate (src, tgt, rel)
        for _, target, data in self.graph.out_edges(src, data=True):
            if data.get("rel") == rel and target == tgt:
                return
        self.graph.add_edge(src, tgt, rel=rel)

    # ---------- Visualization -------------------------------------------------

    def show_graph_streamlit(self):
        """
        Streamlit + PyVis visualization. Uses node 'name' and edge 'rel'.
        """
        if Network is None or st is None:
            raise RuntimeError("pyvis/streamlit not available in this environment.")

        

        net = Network(height="600px", width="100%", directed=True,
                    bgcolor="#FAFAFA", font_color="#202124")
        net.barnes_hut()

        for node, attrs in self.graph.nodes(data=True):
            label = attrs.get("name", str(node))
            color = self.CATEGORY_COLOR.get(attrs.get("kind"), self.CATEGORY_COLOR["_default"])
            net.add_node(
                n_id=node,
                label=label,
                title=json.dumps(attrs, indent=2),
                color=color
            )

        for source, target, attrs in self.graph.edges(data=True):
            edge_label = attrs.get("rel", "")
            net.add_edge(
                source,
                target,
                label=edge_label,
                title=json.dumps(attrs, indent=2),
                color="#888888"
            )

        # Generate HTML for the graph
        html_content = net.generate_html()

        # Show legend (above the graph, in Streamlit)
        legend_items = []
        for kind, color in self.CATEGORY_COLOR.items():
            if kind == "_default":
                continue
            legend_items.append(
                f"<div style='display:inline-block; margin-right:12px;'>"
                f"<span style='display:inline-block; width:14px; height:14px; background:{color}; margin-right:4px;'></span>"
                f"{kind}</div>"
            )
        legend_html = "<div style='margin-bottom:10px;'>" + "".join(legend_items) + "</div>"

        # Combine legend + graph
        combined_html = legend_html + html_content

        st.components.v1.html(combined_html, height=700, scrolling=True)


    def show_graph(self):
        """Quick Matplotlib draw of the entire graph."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        node_colors = [
            self.CATEGORY_COLOR.get(d.get("kind"), self.CATEGORY_COLOR["_default"])
            for _, d in self.graph.nodes(data=True)
        ]
        labels = {n: d.get("name", str(n)) for n, d in self.graph.nodes(data=True)}
        nx.draw(self.graph, pos, with_labels=True, labels=labels,
                node_size=600, font_size=9, node_color=node_colors,
                edgecolors="#333333", linewidths=1.3, arrows=True)
        plt.tight_layout()
        plt.show()

    # ---------- Attribute helpers --------------------------------------------

    def get_attr_keys(self) -> List[str]:
        keys = set()
        for _, d in self.graph.nodes(data=True):
            keys.update(d.keys())
        return sorted(keys)

    def get_attr_values(self, kind: str) -> list:
        # Return list of 'name' values for nodes of a given kind
        print(self.graph.nodes(data=True))
        print(f"kinds got {kind}", [
            data.get("name", str(n))
            for n, data in self.graph.nodes(data=True)
            if data.get("kind") == kind
        ])
        return [
            data.get("name", str(n))
            for n, data in self.graph.nodes(data=True)
            if data.get("kind") == kind
        ]

    def subgraph_for_attr(self, key, value) -> nx.Graph:
        nodes = [n for n, d in self.graph.nodes(data=True) if d.get(key) == value]
        return self.graph.subgraph(nodes).copy()

    def neighbour_subgraph(self, node_id) -> nx.Graph:
        return self.graph.subgraph([node_id, *self.graph.neighbors(node_id)]).copy()

    def get_node_labels(self) -> List[str]:
        kinds = set()
        for _, data in self.graph.nodes(data=True):
            kind = data.get("kind")
            if kind:
                kinds.add(kind)
        return sorted(kinds)

    # ---------- Relations / Augmentation -------------------------------------

    def fetch_related(self, node_name: str) -> List[Dict]:
        """
        Fetch neighbors of a given node by its 'name'.

        Returns list of dicts with:
        - neighbour: neighbor node 'name'
        - rel_type: edge 'rel'
        - direction: 'out' (node_name → neighbor) or 'in' (neighbor → node_name)
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
            related.append({"neighbour": neighbor_name, "rel_type": rel_type, "direction": "out"})

        # Incoming edges
        for source, _, edge_data in self.graph.in_edges(target_id, data=True):
            neighbor_name = self.graph.nodes[source].get("name", str(source))
            rel_type = edge_data.get("rel", "related_to")
            related.append({"neighbour": neighbor_name, "rel_type": rel_type, "direction": "in"})

        related.sort(key=lambda x: (x["rel_type"], x["neighbour"]))
        return related

    def augment_query_with_graph(self, query_terms: List[str], depth: int = 2) -> List[str]:
        """
        Expand query terms via multi-hop neighborhood based on node 'name',
        skipping nodes of type 'Paper'.
        """
        name_to_nodes = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            name = data.get("name")
            if name and name.lower() in query_terms:
                name_to_nodes[name].append(node)

        expanded = set(query_terms)
        for term in query_terms:
            matching_nodes = name_to_nodes.get(term, [])
            for node_id in matching_nodes:
                if node_id in self.graph and self.graph.nodes[node_id].get("kind") in {"Keyword", "MetaData"}:
                    for neighbor_id in nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth):
                        neighbor_data = self.graph.nodes[neighbor_id]
                        if neighbor_data.get("kind") != "Paper":
                            nname = neighbor_data.get("name")
                            if nname:
                                expanded.add(nname)
        return list(expanded)

    # ---------- Pretty subgraph drawing / interactive -------------------------

    #def _color_for_kind(self, kind: str) -> str:
    #    return self.CATEGORY_COLOR.get(kind, self.CATEGORY_COLOR["_default"])

    def _subgraph_attributes_df(self, H: nx.Graph):
        """Collect an attributes table for nodes shown in subgraph H."""
        rows = []
        for n in H.nodes():
            # We build subgraph from human-readable names in interactive view,
            # map back to original node by matching 'name'.
            base_id = None
            node_attrs = None
            for nid, attrs in self.graph.nodes(data=True):
                if attrs.get("name") == n:
                    base_id = nid
                    node_attrs = attrs
                    break
            if base_id is None:
                base_id = n
                node_attrs = self.graph.nodes.get(n, {})

            row = {"node_id": base_id}
            row.update(node_attrs)
            rows.append(row)

        if pd is not None:
            df = pd.DataFrame(rows)
            preferred = ["node_id", "name", "kind", "layer", "category", "unit", "value"]
            cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
            return df[cols]
        return rows

    def draw_neighbour_subgraph_by_name(self, node_name: str):
        """Non-interactive: draw 1-hop neighborhood around a node name."""
        target_id = None
        for nid, data in self.graph.nodes(data=True):
            if data.get("name") == node_name:
                target_id = nid
                break
        if target_id is None:
            print(f"Node with name '{node_name}' not found.")
            return

        H = self.graph.subgraph([target_id, *self.graph.neighbors(target_id)]).copy()
        labels = {n: self.graph.nodes[n].get("name", str(n)) for n in H.nodes()}
        colors = [self._color_for_kind(self.graph.nodes[n].get("kind")) for n in H.nodes()]
        sizes = [600 + 160 * H.degree(n) for n in H.nodes()]

        plt.figure(figsize=(9, 7), dpi=140)
        pos = nx.spring_layout(H, seed=42)
        nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colors,
                               edgecolors="#333", linewidths=1.8)
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=10, font_weight="semibold")
        nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="-|>", arrowsize=16,
                               width=1.4, alpha=0.85, connectionstyle="arc3,rad=0.12")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def show_subgraph_interactive(self):
        """
        Jupyter-only: interactively explore neighbors around a selected node.
        - Pick a node kind (Layer/Category/Keyword/Paper/MetaData)
        - Pick a node by its 'name'
        - See a small neighbor graph (colored by kind) + table of attributes
        """
        if widgets is None or display is None:
            raise RuntimeError("Interactive widgets not available in this environment.")

        node_labels = self.get_node_labels()
        if not node_labels:
            print("No node labels found in the knowledge graph.")
            return

        # Dropdowns
        label_dropdown = widgets.Dropdown(options=node_labels, description="Node type:")
        values = self.get_attr_values(node_labels[0])
        value_dropdown = widgets.Dropdown(options=values, description="Value:")
        show_edge_labels_cb = widgets.Checkbox(value=True, description="Show edge labels")

        # Render controls
        box = widgets.VBox([label_dropdown, value_dropdown, show_edge_labels_cb])
        display(box)

        def on_label_change(change):
            new_values = self.get_attr_values(change["new"])
            value_dropdown.options = new_values

        label_dropdown.observe(on_label_change, names="value")

        def on_value_change(change):
            choice = change["new"]
            clear_output(wait=True)
            display(box)

            if not choice:
                return

            data = self.fetch_related(choice)
            print(f"Neighbours of {choice}")
            if not data:
                print("No neighbors.")
                return

            # Attributes table for nodes in this subgraph (after building G)
            # Build tiny name-based DiGraph for visualization
            G = nx.DiGraph()
            G.add_node(choice)
            for row in data:
                n = row["neighbour"]
                rel = row["rel_type"]
                if row["direction"] == "out":
                    G.add_edge(choice, n, label=rel)
                else:
                    G.add_edge(n, choice, label=rel)

            # Show table
            attrs_df = self._subgraph_attributes_df(G)
            if pd is not None:
                display(attrs_df)
            else:
                print(attrs_df)

            # Colors/sizes/labels
            node_colors = []
            node_sizes = []
            node_labels_local = {}
            for node in G.nodes():
                # map back to original node by name
                base_id = None
                for nid, attrs in self.graph.nodes(data=True):
                    if attrs.get("name") == node:
                        base_id = nid
                        node_kind = attrs.get("kind")
                        break
                if base_id is None:
                    node_kind = self.graph.nodes.get(node, {}).get("kind")

                color = self._color_for_kind(node_kind)
                node_colors.append(color)
                node_sizes.append(600 + 160 * G.degree(node))
                node_labels_local[node] = node

            # Draw nicely
            plt.close("all")
            plt.figure(figsize=(9.5, 7.2), dpi=140)
            pos = nx.spring_layout(G, k=1.1, seed=42)

            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                linewidths=1.8, edgecolors="#333333", alpha=0.95
            )
            nx.draw_networkx_labels(
                G, pos, labels=node_labels_local,
                font_size=10, font_weight="semibold"
            )
            nx.draw_networkx_edges(
                G, pos,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=16,
                width=1.4,
                alpha=0.85,
                connectionstyle="arc3,rad=0.12"
            )

            if show_edge_labels_cb.value:
                edge_labels = nx.get_edge_attributes(G, "label")
                nx.draw_networkx_edge_labels(
                    G, pos, edge_labels=edge_labels,
                    font_size=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                    rotate=False
                )

            # Legend
            handles = []
            for kind, color in self.CATEGORY_COLOR.items():
                if kind == "_default":
                    continue
                handles.append(mpatches.Patch(color=color, label=kind))
            if handles:
                plt.legend(handles=handles, title="Node kind",
                           frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

            plt.axis("off")
            plt.tight_layout()
            plt.show()

        value_dropdown.observe(on_value_change, names="value")
    

    def generate_nodes_html(self, node_type: str) -> str:
        """
        Return PyVis HTML containing *only nodes* (no edges).

        Parameters
        ----------
        node_type : str
            One of "All", "Layer", "Category", "Keyword", "Paper", "MetaData".
        """
        

        net = Network(height="600px", width="100%", directed=False,
                    bgcolor="#FAFAFA", font_color="#202124")
        net.barnes_hut()  # layout

        for node_id, data in self.graph.nodes(data=True):
            label = data.get("kind", "Unknown")

            if node_type == "All" or label == node_type:
                name = data.get("name", str(node_id))
                color = self.CATEGORY_COLOR.get(label, self.CATEGORY_COLOR["_default"])
                net.add_node(node_id, label=name, color=color, title=label)

        return net.generate_html()

    # ---------- Pretty subgraph drawing / interactive -------------------------

    #def _color_for_kind(self, kind: str) -> str:
    #    return self.CATEGORY_COLOR.get(kind, self.CATEGORY_COLOR["_default"])

    def _subgraph_attributes_df(self, H: nx.Graph):
        """Collect an attributes table for nodes shown in subgraph H."""
        rows = []
        for n in H.nodes():
            # We build subgraph from human-readable names in interactive view,
            # map back to original node by matching 'name'.
            base_id = None
            node_attrs = None
            for nid, attrs in self.graph.nodes(data=True):
                if attrs.get("name") == n:
                    base_id = nid
                    node_attrs = attrs
                    break
            if base_id is None:
                base_id = n
                node_attrs = self.graph.nodes.get(n, {})

            row = {"node_id": base_id}
            row.update(node_attrs)
            rows.append(row)

        if pd is not None:
            df = pd.DataFrame(rows)
            preferred = ["node_id", "name", "kind", "layer", "category", "unit", "value"]
            cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
            return df[cols]
        return rows

    def draw_neighbour_subgraph_by_name(self, node_name: str):
        """Non-interactive: draw 1-hop neighborhood around a node name."""
        target_id = None
        for nid, data in self.graph.nodes(data=True):
            if data.get("name") == node_name:
                target_id = nid
                break
        if target_id is None:
            print(f"Node with name '{node_name}' not found.")
            return

        H = self.graph.subgraph([target_id, *self.graph.neighbors(target_id)]).copy()
        labels = {n: self.graph.nodes[n].get("name", str(n)) for n in H.nodes()}
        colors = [self._color_for_kind(self.graph.nodes[n].get("kind")) for n in H.nodes()]
        sizes = [600 + 160 * H.degree(n) for n in H.nodes()]

        plt.figure(figsize=(9, 7), dpi=140)
        pos = nx.spring_layout(H, seed=42)
        nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colors,
                               edgecolors="#333", linewidths=1.8)
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=10, font_weight="semibold")
        nx.draw_networkx_edges(H, pos, arrows=True, arrowstyle="-|>", arrowsize=16,
                               width=1.4, alpha=0.85, connectionstyle="arc3,rad=0.12")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def show_subgraph_interactive(self):
        """
        Jupyter-only: interactively explore neighbors around a selected node.
        - Pick a node kind (Layer/Category/Keyword/Paper/MetaData)
        - Pick a node by its 'name'
        - See a small neighbor graph (colored by kind) + table of attributes
        """
        if widgets is None or display is None:
            raise RuntimeError("Interactive widgets not available in this environment.")

        node_labels = self.get_node_labels()
        if not node_labels:
            print("No node labels found in the knowledge graph.")
            return

        # Dropdowns
        label_dropdown = widgets.Dropdown(options=node_labels, description="Node type:")
        values = self.get_attr_values(node_labels[0])
        value_dropdown = widgets.Dropdown(options=values, description="Value:")
        show_edge_labels_cb = widgets.Checkbox(value=True, description="Show edge labels")

        # Render controls
        box = widgets.VBox([label_dropdown, value_dropdown, show_edge_labels_cb])
        display(box)

        def on_label_change(change):
            new_values = self.get_attr_values(change["new"])
            value_dropdown.options = new_values

        label_dropdown.observe(on_label_change, names="value")

        def on_value_change(change):
            choice = change["new"]
            clear_output(wait=True)
            display(box)

            if not choice:
                return

            data = self.fetch_related(choice)
            print(f"Neighbours of {choice}")
            if not data:
                print("No neighbors.")
                return

            # Attributes table for nodes in this subgraph (after building G)
            # Build tiny name-based DiGraph for visualization
            G = nx.DiGraph()
            G.add_node(choice)
            for row in data:
                n = row["neighbour"]
                rel = row["rel_type"]
                if row["direction"] == "out":
                    G.add_edge(choice, n, label=rel)
                else:
                    G.add_edge(n, choice, label=rel)

            # Show table
            attrs_df = self._subgraph_attributes_df(G)
            if pd is not None:
                display(attrs_df)
            else:
                print(attrs_df)

            # Colors/sizes/labels
            node_colors = []
            node_sizes = []
            node_labels_local = {}
            for node in G.nodes():
                # map back to original node by name
                base_id = None
                for nid, attrs in self.graph.nodes(data=True):
                    if attrs.get("name") == node:
                        base_id = nid
                        node_kind = attrs.get("kind")
                        break
                if base_id is None:
                    node_kind = self.graph.nodes.get(node, {}).get("kind")

                color = self._color_for_kind(node_kind)
                node_colors.append(color)
                node_sizes.append(600 + 160 * G.degree(node))
                node_labels_local[node] = node

            # Draw nicely
            plt.close("all")
            plt.figure(figsize=(9.5, 7.2), dpi=140)
            pos = nx.spring_layout(G, k=1.1, seed=42)

            nx.draw_networkx_nodes(
                G, pos,
                node_size=node_sizes,
                node_color=node_colors,
                linewidths=1.8, edgecolors="#333333", alpha=0.95
            )
            nx.draw_networkx_labels(
                G, pos, labels=node_labels_local,
                font_size=10, font_weight="semibold"
            )
            nx.draw_networkx_edges(
                G, pos,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=16,
                width=1.4,
                alpha=0.85,
                connectionstyle="arc3,rad=0.12"
            )

            if show_edge_labels_cb.value:
                edge_labels = nx.get_edge_attributes(G, "label")
                nx.draw_networkx_edge_labels(
                    G, pos, edge_labels=edge_labels,
                    font_size=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                    rotate=False
                )

            # Legend
            handles = []
            for kind, color in self.CATEGORY_COLOR.items():
                if kind == "_default":
                    continue
                handles.append(mpatches.Patch(color=color, label=kind))
            if handles:
                plt.legend(handles=handles, title="Node kind",
                           frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

            plt.axis("off")
            plt.tight_layout()
            plt.show()

        value_dropdown.observe(on_value_change, names="value")
    

    def generate_nodes_html(self, node_type: str) -> str:
        """
        Return PyVis HTML containing *only nodes* (no edges).

        Parameters
        ----------
        node_type : str
            One of "All", "Layer", "Category", "Keyword", "Paper", "MetaData".
        """
        

        net = Network(height="600px", width="100%", directed=False,
                    bgcolor="#FAFAFA", font_color="#202124")
        net.barnes_hut()  # layout

        for node_id, data in self.graph.nodes(data=True):
            label = data.get("kind", "Unknown")

            if node_type == "All" or label == node_type:
                name = data.get("name", str(node_id))
                color = self.CATEGORY_COLOR.get(label, self.CATEGORY_COLOR["_default"])
                net.add_node(node_id, label=name, color=color, title=label)

        return net.generate_html()
    def get_node_attr(self, node_name: str, attr_key: str):
        """
        Return a specific attribute (attr_key) of a node identified by its 'name' attribute.
        Returns None if the node or attribute is not found.
        """
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get("name") == node_name:
                return attrs.get(attr_key)
        return None