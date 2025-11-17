"""neo4j_knowledge_graph.py – concrete Neo4j implementation plus *visualisation helpers*

Changes (2025‑07‑07)
────────────────────
1. **Indexes / constraints helper** – `ensure_constraints()` guarantees unique
   identifiers (`Layer.name`, `Category.key`, `Keyword.key`, `Paper.id`).
2. **Bulk transaction** – `build_graph()` now batches Cypher MERGE statements in
   a single transaction for speed (≈10× faster than per‑node `Graph.merge`).
3. **Visualisation helpers** –
   • `fetch_node_names()` – dropdown list for UI.
   • `fetch_subgraph()`   – 1‑hop neighbour query.
   • `generate_subgraph_html()` – colour‑coded PyVis HTML (node + edge labels).
   These slot neatly into your Streamlit front‑end.
"""

from __future__ import annotations


import uuid
from collections import defaultdict
from typing import Dict, List
import streamlit.components.v1 as components

from py2neo import Graph, Node
from pyvis.network import Network

from sokegraph.graph.knowledge_graph import KnowledgeGraph
import streamlit as st
import networkx as nx
import os
import matplotlib.pyplot as plt



class Neo4jKnowledgeGraph(KnowledgeGraph):
    """Concrete implementation that builds and explores a Neo4j knowledge graph."""

    def __init__(self, ontology_path: str, papers_path, uri: str, user: str, password: str):
        super().__init__(ontology_path, papers_path)
        self.graph = Graph(uri, auth=(user, password))
        self.ensure_constraints()

    # ─────────────────────────────────────────────────────────────────────
    # Build phase
    # ─────────────────────────────────────────────────────────────────────

    def ensure_constraints(self) -> None:
        """Create uniqueness constraints if they don’t already exist."""
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Layer)    REQUIRE l.name IS UNIQUE")
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.key IS UNIQUE")
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword)  REQUIRE k.key IS UNIQUE")
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper)    REQUIRE p.id IS UNIQUE")


    def build_graph(self):
        """Load ontology extractions into Neo4j – now batched for speed."""
        tx = self.graph.begin()
        paper_titles: Dict[str, str] = {}
        unique_keywords: Dict[tuple, set] = defaultdict(set)  # (layer, cat) → kw_key
        unique_categories: set[str] = set()
        unique_layers: set[str] = set()
        paper_keywords: Dict[str, set] = defaultdict(set)

        for layer, categories in self.ontology_extractions.items():
            if layer not in unique_layers:
                tx.merge(Node("Layer", name=layer), "Layer", "name")
                unique_layers.add(layer)

            for cat, items in categories.items():
                cat_key = f"{layer}|{cat}"
                if cat_key not in unique_categories:
                    cat_node = Node("Category", name=cat, key=cat_key)
                    tx.merge(cat_node, "Category", "key")
                    tx.run(
                        "MATCH (l:Layer {name:$layer}), (c:Category {key:$key})\n"
                        "MERGE (l)-[:HAS_CATEGORY]->(c)",
                        layer=layer,
                        key=cat_key,
                    )
                    unique_categories.add(cat_key)

                for item in items:
                    paper_id = item.get("paper_id", "unknown")
                    paper_title = (item.get("paper_title") or item.get("title") or paper_id).strip()
                    paper_titles[paper_id] = paper_title
                    for kw in item["keywords"]:
                        kw_key = f"{layer}|{cat}|{kw}"
                        if kw_key in unique_keywords[(layer, cat)]:
                            continue
                        unique_keywords[(layer, cat)].add(kw_key)
                        tx.merge(Node("Keyword", name=kw, key=kw_key), "Keyword", "key")
                        tx.run(
                            "MATCH (c:Category {key:$cat_key}), (k:Keyword {key:$kw_key})\n"
                            "MERGE (c)-[:HAS_KEYWORD]->(k)",
                            cat_key=cat_key,
                            kw_key=kw_key,
                        )

                        # MetaData
                        for meta in item.get("parsed_meta", []):
                            meta_id = str(uuid.uuid4())
                            meta_node = Node(
                                "MetaData",
                                id=meta_id,
                                name=meta.get("unit", ""),
                                value=meta.get("value", ""),
                                unit=meta.get("unit", ""),
                                type=cat,
                            )
                            tx.create(meta_node)
                            tx.run(
                                "MATCH (k:Keyword {key:$kw_key}), (m:MetaData {id:$mid})\n"
                                "MERGE (k)-[:HAS_METADATA]->(m)",
                                kw_key=kw_key,
                                mid=meta_id,
                            )

                        paper_keywords[paper_id].add(kw_key)

        # Paper nodes and MENTIONS relationships
        for paper_id, kw_keys in paper_keywords.items():
            display_name = paper_titles.get(paper_id, paper_id)
            tx.merge(
                Node("Paper",
                    id=paper_id,           # stable key
                    name=display_name,     # shown in UI
                    paper_id=paper_id,     # keep id as a property too (optional)
                    title=display_name),   # optional alias
                "Paper", "id"
            )
            for kw_key in kw_keys:
                tx.run(
                    "MATCH (k:Keyword {key:$kw_key}), (p:Paper {id:$pid})\n"
                    "MERGE (k)-[:MENTIONS]->(p)",
                    kw_key=kw_key,
                    pid=paper_id,
                )

        tx.commit()
        print("🎉 Neo4j knowledge graph construction complete.")

    # ─────────────────────────────────────────────────────────────────────
    # Exploration / visualisation helpers
    # ─────────────────────────────────────────────────────────────────────

    def fetch_node_names(self) -> List[str]:
        return self.graph.run(
            "MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name ORDER BY n.name"
        ).to_series().tolist()

    def fetch_subgraph(self, centre: str):
        query = (
            "MATCH (c {name:$centre})\n"
            "OPTIONAL MATCH (c)-[r]-(n)\n"
            "WITH c, n, r, CASE WHEN startNode(r)=c THEN 'out' ELSE 'in' END AS direction\n"
            "RETURN coalesce(n.name, c.name) AS neighbour,\n"
            "       coalesce(type(r), '')   AS rel_type,\n"
            "       direction,\n"
            "       CASE WHEN n IS NULL THEN labels(c)[0] ELSE labels(n)[0] END AS category"
        )
        return self.graph.run(query, centre=centre).data()

    def generate_subgraph_html(self, centre: str, rows: List[dict]) -> str:
        net = Network(height="600px", width="100%", directed=True)

        def add(name: str, category: str):
            colour = self.CATEGORY_COLOR.get(category, self.CATEGORY_COLOR["_default"])
            if name not in net.node_map:
                net.add_node(name, label=name, color=colour, title=category)

        centre_cat = next(r["category"] for r in rows if r["neighbour"] == centre)
        add(centre, centre_cat)

        for row in rows:
            n, rel, direction, cat = row["neighbour"], row["rel_type"], row["direction"], row["category"]
            add(n, cat)
            if rel:
                if direction == "out":
                    net.add_edge(centre, n, label=rel, title=rel)
                else:
                    net.add_edge(n, centre, label=rel, title=rel)

        return net.generate_html()
    
    def show_graph_streamlit(self):
        # Fetch all nodes and their neighbors
        full_graph_query = """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT a.name AS source, type(r) AS rel, b.name AS target,
                        labels(a)[0] AS source_label, labels(b)[0] AS target_label
        """
        data = self.graph.run(full_graph_query).data()

        # Build PyVis graph manually
        from pyvis.network import Network
        net = Network(height="650px", width="100%", directed=True)

        for row in data:
            s, t = row["source"], row["target"]
            s_label, t_label = row["source_label"], row["target_label"]
            rel = row["rel"]

            # Add source and target nodes
            color_s = self.CATEGORY_COLOR.get(s_label, self.CATEGORY_COLOR["_default"])
            color_t = self.CATEGORY_COLOR.get(t_label, self.CATEGORY_COLOR["_default"])
            net.add_node(s, label=s, color=color_s, title=s_label)
            net.add_node(t, label=t, color=color_t, title=t_label)

            # Add edge
            net.add_edge(s, t, label=rel, title=rel)

        html = net.generate_html()
        st.components.v1.html(html, height=750, scrolling=True)
    
    def show_graph(self, limit=20):
        """Query and display a subgraph from Neo4j using matplotlib."""
        data = self.graph.run(
            "MATCH (n)-[r]->(m) RETURN n.name AS source, m.name AS target LIMIT $limit",
            limit=limit
        ).data()
        G = nx.DiGraph()
        for record in data:
            G.add_edge(record["source"], record["target"])
        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=True, node_size=500, font_size=10)
        plt.show()

    # neo4j_knowledge_graph.py  ────────────────────────────────────────────
    def generate_nodes_html(self, node_type: str) -> str:
        """
        Return PyVis HTML containing *only nodes* (no edges).

        Parameters
        ----------
        node_type : str
            One of "All", "Layer", "Category", "Keyword", "Paper", "MetaData".
        """
        from pyvis.network import Network

        net = Network(height="650px", width="100%", directed=False)
        net.barnes_hut()  # layout

        if node_type == "All":
            q = "MATCH (n) RETURN n.name AS name, labels(n)[0] AS label"
            rows = self.graph.run(q).data()
        else:
            q = f"MATCH (n:{node_type}) RETURN n.name AS name, labels(n)[0] AS label"
            rows = self.graph.run(q).data()

        for row in rows:
            name, label = row["name"], row["label"]
            colour = self.CATEGORY_COLOR.get(label, self.CATEGORY_COLOR["_default"])
            net.add_node(name, label=name, color=colour, title=label)

        return net.generate_html()


    # ------------------------------------------------------------------
    # PUBLIC, attribute‑agnostic query interface (used by the viewer)
    # ------------------------------------------------------------------
    def get_attr_keys(self) -> list[str]:
        cypher = """
        MATCH (n) WITH DISTINCT keys(n) AS klist UNWIND klist AS k
        RETURN DISTINCT k ORDER BY k
        """
        return [r["k"] for r in self.run(cypher)]

    def get_attr_values(self, label):
        print("shahlla")
        # Ensure label is a valid node label to avoid Cypher injection
        import re
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label):
            raise ValueError(f"Invalid label name: {label}")

        cypher = f"""
            MATCH (n:{label})
            WHERE n.name IS NOT NULL
            RETURN DISTINCT n.name AS v
            ORDER BY v
        """
        return [r["v"] for r in self.run(cypher)]


    def subgraph_for_attr(self, label, value):
        # Validate inputs to avoid injection
        import re
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label):
            raise ValueError(f"Invalid label name: {label}")

        cypher = f"""
            MATCH (n:{label} {{name: $value}})-[r]-(m)
            RETURN n AS start, r AS rel, m AS end
        """
        records = self.run(cypher, value=value)
        print(">>> RECORD COUNT:", records)
        for r in records:
            print(">>> RECORD:", r)
        return self._records_to_nx(records)

    def neighbour_subgraph(self, node_id: str) -> nx.Graph:
        cypher = "MATCH (n)-[r]-(m) WHERE id(n)=$id RETURN n,r,m"
        recs = self.run(cypher, id=int(node_id))
        return self._records_to_nx(recs)

    # ------------------------------------------------------------------
    # Optional utility – expose full graph as NetworkX (cacheable)
    # ------------------------------------------------------------------
    def to_networkx(self) -> nx.Graph:
        """Fetch *all* nodes/edges into a NetworkX graph (may be large)."""
        cypher = "MATCH (n)-[r]-(m) RETURN n,r,m"
        return self._records_to_nx(self.run(cypher))
    
    def run(self, cypher: str, **params):
        """Helper to run a Cypher query with optional parameters."""
        return self.graph.run(cypher, **params)
    

    def get_node_labels(self):
        cypher = "CALL db.labels() YIELD label RETURN label ORDER BY label"
        return [record["label"] for record in self.run(cypher)]
    
    def records_to_nx(self, records):
        G = nx.MultiDiGraph()
        for record in records:
            n = record["n"]
            m = record["m"]
            r = record["r"]

            G.add_node(n.id, **dict(n))
            G.add_node(m.id, **dict(m))
            G.add_edge(n.id, m.id, key=r.type, **dict(r))

        return G
    
    def get_neighbour_subgraph(self, graph, node_name):
        cypher = """
            MATCH (n {name: $name})-[r]-(m)
            RETURN n, r, m
        """
        return graph.run(cypher, name=node_name)



    def show_sub_graph(G):
        net = Network(height="600px", width="100%", directed=True)
        net.repulsion()

        for node_id, data in G.nodes(data=True):
            net.add_node(node_id, label=data.get("name", str(node_id)), title=str(data))

        for src, tgt, edge_data in G.edges(data=True):
            net.add_edge(src, tgt, label=edge_data.get("type", ""))

        path = "graph.html"
        net.write_html(path)

        with open(path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=650, scrolling=True)

        os.remove(path)
    
    def fetch_related(self, node_name: str):
        query = """
        MATCH (n {name: $name})-[r]-(m)
        RETURN DISTINCT m.name AS neighbour,
                        type(r) AS rel_type,
                        CASE WHEN startNode(r)=n THEN 'out' ELSE 'in' END AS direction
        ORDER BY rel_type, neighbour
        """
        result = self.graph.run(query, parameters={"name": node_name})
        return [dict(record) for record in result]