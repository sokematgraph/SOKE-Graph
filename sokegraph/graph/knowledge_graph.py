"""
knowledge_graph.py

Defines an abstract base class for all knowledge-graph builders used in
SOKEGraph (e.g., Neo4jGraph, NetworkXGraph, etc.).  Subclasses must
implement `build_graph`, which takes the loaded ontology extractions and
creates the actual graph in the chosen backend.
"""

from abc import ABC, abstractmethod
import json
from typing import Any, Optional, Set, Dict, List
import streamlit as st
from pyvis.network import Network
from sokegraph.utils.functions import wrap_label
import pandas as pd
from sokegraph.utils.functions import load_papers


class KnowledgeGraph(ABC):

    CATEGORY_COLOR: Dict[str, str] = {
    "Layer": "#FFD700",    # gold
    "Category": "#FF7F50", # coral
    "Keyword": "#87CEFA",  # light sky blue
    "Paper": "#90EE90",    # light green
    "MetaData": "#D3D3D3", # light gray
    "_default": "#CCCCCC"  # fallback
    }   
    """Abstract base class for building a knowledge graph from an ontology.

    Attributes
    ----------
    ontology_path : str
        Path to the ontology-extraction JSON file.
    ontology_extractions : dict
        Parsed JSON loaded from `ontology_path`, ready for graph construction.
    """

    # ------------------------------------------------------------------ #
    # Constructor & helpers                                              #
    # ------------------------------------------------------------------ #
    def __init__(self, ontology_path: str, papers_path:str) -> None:
        """
        Load ontology extractions from disk and store them for later use.

        Parameters
        ----------
        ontology_path : str
            Path to the JSON file produced by the extraction pipeline.
        """
        self.ontology_path = ontology_path
        self.ontology_extractions = self._load_ontology()
        self.papers = load_papers(papers_path)

    def _display_label(self, node_key_or_name: str) -> str:
        """Return the label to show in PyVis (prefers graph 'name', falls back to key)."""
        human = self._get_attr(node_key_or_name, "name")
        return wrap_label(human if human else str(node_key_or_name))

    def _get_attr(self, node_key_or_name: str, key: str):
        """Get a node attribute (e.g., 'name' or 'kind').
        Works whether you pass the internal node key or the human display name."""
        G = self.graph

        # 1) If it's an actual node key
        if node_key_or_name in G:
            return G.nodes[node_key_or_name].get(key)

        # 2) If caller passed a display name, resolve by name→node
        for nid, attrs in G.nodes(data=True):
            if attrs.get("name") == node_key_or_name:
                return attrs.get(key)

        return None

    def _display_label(self, node_key_or_name: str) -> str:
        """Return the label to show in PyVis (prefers graph 'name', falls back to key)."""
        human = self._get_attr(node_key_or_name, "name")
        # use your existing wrap_label (function or method)
        
        return wrap_label(human if human else str(node_key_or_name))
    
    def _load_ontology(self) -> Any:
        """Read the ontology JSON into a Python object."""
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            return json.load(f)
        

    




    def _color_for_kind(self, kind) -> str:
        if not kind:
            return self.CATEGORY_COLOR.get("_default", "#CCCCCC")
        return self.CATEGORY_COLOR.get(kind, self.CATEGORY_COLOR.get("_default", "#CCCCCC"))

    def _node_style(self, *, kind, is_central: bool = False) -> dict:
        c = self._color_for_kind(kind)
        return {
            "shape": "dot",
            "size": 26 if is_central else 18,
            "font": {"color": "#ffffff", "size": 16 if is_central else 14},
            "color": {"background": c, "border": c},
            "borderWidth": 2 if is_central else 1,
        }

    def _node_kind(self, node: str) -> str:
        try:
            return self.get_node_attr(node, "kind")
        except Exception:
            return None

    def _legend_html(self, kinds) -> str:
        uniq = sorted({k for k in kinds if k})
        if not uniq:
            return ""
        chips = []
        for k in uniq:
            chips.append(
                f'<span style="display:inline-flex;align-items:center;gap:8px;'
                f'padding:4px 8px;border:1px solid #e5e7eb;border-radius:9999px;'
                f'margin:4px 6px 0 0;font:13px/1.2 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;">'
                f'<span style="width:12px;height:12px;border-radius:50%;background:{self._color_for_kind(k)};"></span>{k}'
                f'</span>'
            )
        return '<div style="margin:8px 0 4px 0;">' + "".join(chips) + "</div>"

    def show_subgraph_streamlit(self, choice):
        import streamlit as st
        from pyvis.network import Network
        import json, tempfile, uuid
        from pathlib import Path

        kg = st.session_state.get("kg_result", self)

        data = kg.fetch_related(choice)
        st.subheader(f"Neighbours of **{choice}**")
        st.dataframe(data, use_container_width=True)

        if not st.checkbox("Visualise neighbours (PyVis)"):
            return

        # ---------- name → kind index (for coloring) ----------
        name_to_kind = {}
        try:
            for nid, attrs in kg.graph.nodes(data=True):
                nm = attrs.get("name", nid)
                k  = attrs.get("kind")
                if nm is not None:
                    name_to_kind[str(nm)] = k
                name_to_kind[str(nid)] = k
        except Exception:
            pass

        def _normalize_kind(k):
            if k is None:
                return None
            ks = str(k).strip().lower()
            for key in self.CATEGORY_COLOR.keys():
                if key != "_default" and key.lower() == ks:
                    return key
            return None


        # ---------------- build PyVis ----------------

        net = Network(height="600px", width="100%", directed=True,
                    bgcolor="#FAFAFA", font_color="#202124")

        net.add_node(choice, label=self._display_label(choice), **self.style_for(choice, is_center=True))

        for row in data:
            n    = row["neighbour"]
            rel  = row.get("rel_type", "")
            dirn = row.get("direction", "out")

            if n not in net.node_map:
                net.add_node(n, label=self._display_label(n), **self.style_for(n))

            if dirn == "out":
                net.add_edge(choice, n, label=rel, title=rel, color="#888888")
            else:
                net.add_edge(n, choice, label=rel, title=rel, color="#888888")

        net.repulsion(node_distance=180, central_gravity=0.2,
                    spring_length=200, spring_strength=0.05, damping=0.09)

        # --------- Legend (Streamlit HTML above the graph) ----------
        legend_items = []
        for kind, color in self.CATEGORY_COLOR.items():
            if kind == "_default":
                continue
            legend_items.append(
                f"""
                <div class="legend-item">
                <span class="swatch" style="background:{color}"></span>
                <span class="label">{kind}</span>
                </div>
                """
            )

        legend_html = f"""
        <style>
        .legend-wrap {{
            display: flex; flex-wrap: wrap; gap: 10px;
            align-items: center; margin: 6px 0 10px 0;
            font-size: 14px;
        }}
        .legend-item {{ display:flex; align-items:center; gap:6px; }}
        .legend-item .swatch {{
            width: 14px; height: 14px; border-radius:3px; border:1px solid rgba(0,0,0,0.15);
            display:inline-block;
        }}
        .legend-edge {{ margin-left: 14px; color:#666; }}
        .legend-edge code {{ background:#f3f3f3; padding:1px 4px; border-radius:4px; }}
        </style>
        <div class="legend-wrap">
        {''.join(legend_items)}
        <span class="legend-edge">Edges: <code>source → target</code>, label = <em>rel_type</em></span>
        </div>
        """

        # Render legend + graph together
        html_content = net.generate_html()
        st.components.v1.html(legend_html + html_content, height=800, scrolling=True)

        # Optional debug
        # if st.checkbox("Show resolved kinds (debug)"):
        #     resolved = {choice: name_to_kind.get(str(choice))}
        #     for row in data:
        #         n = row["neighbour"]
        #         resolved[n] = row.get("kind") or name_to_kind.get(str(n))
        #     st.write(resolved)

        

    def style_for(self, node_name: str, is_center: bool = False) -> dict:
        kind = self._node_kind_from(node_name)
        print(f"Node '{node_name}' has kind '{kind}'")
        hex_color = self.CATEGORY_COLOR.get(kind, self.CATEGORY_COLOR.get(kind))
        print(f"Using color '{hex_color}' for kind '{kind}'")
        return dict(
            shape="circle",
            font={"color": "#ffffff", "size": 16 if is_center else 14},
            color={"background": hex_color, "border": hex_color},
            borderWidth=2
        )

    def _canonical_kind_key(self, kind) -> Optional[str]:
        """ Map raw kind (any case/spelling) to a CATEGORY_COLOR key or None if unknown."""
        if kind is None:
            return None
        ks = str(kind).strip().lower()
        for key in self.CATEGORY_COLOR.keys():
            if key == "_default":
                continue
            if key.lower() == ks:
                return key
        return None

    """
    These are the helpers to truncate. 
    """
    def _truncate(self, text: str, max_len: int = 36) -> str:
        if text is None:
            return ""
        s = str(text)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    def _short_label(self, node_id: str) -> str:
        """
        Short, wrapped label for nodes.
        Uses a tighter limit for Papers so titles don’t blow up the layout.
        """
        name = self._get_attr(node_id, "name") or str(node_id)
        kind = self._canonical_kind_key(self._node_kind_from(node_id)) or ""
        limit = 32 if kind == "Paper" else 28
        return wrap_label(self._truncate(name, limit))

    def _node_title_json(self, node_id: str) -> str:
        """
        Metadata shown in the tooltip (like your full-graph view).
        """
        try:
            attrs = dict(self.graph.nodes[node_id])
        except Exception:
            attrs = {}
        # Ensure 'kind' is present for clarity
        attrs.setdefault("kind", self._node_kind_from(node_id))
        # Put a human name up top if missing
        attrs.setdefault("name", self._get_attr(node_id, "name") or str(node_id))
        return json.dumps(attrs, indent=2, ensure_ascii=False)

    def _node_style(self, *, kind, is_central: bool = False) -> dict:
        c = self._color_for_kind(kind)
        return {
            "shape": "dot",
            "size": 26 if is_central else 18,
            "font": {"color": "#ffffff", "size": 16 if is_central else 14},
            "color": {"background": c, "border": c},
            "borderWidth": 2 if is_central else 1,
        }

    def _node_kind(self, node: str) -> str:
        try:
            return self.get_node_attr(node, "kind")
        except Exception:
            return None

    def _legend_html(self, kinds) -> str:
        uniq = sorted({k for k in kinds if k})
        if not uniq:
            return ""
        chips = []
        for k in uniq:
            chips.append(
                f'<span style="display:inline-flex;align-items:center;gap:8px;'
                f'padding:4px 8px;border:1px solid #e5e7eb;border-radius:9999px;'
                f'margin:4px 6px 0 0;font:13px/1.2 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;">'
                f'<span style="width:12px;height:12px;border-radius:50%;background:{self._color_for_kind(k)};"></span>{k}'
                f'</span>'
            )
        return '<div style="margin:8px 0 4px 0;">' + "".join(chips) + "</div>"

    # ------------------------------------------------------------------ #
    # Multi-choice related fetching & visualization                      #
    # ------------------------------------------------------------------ #
    def fetch_related_list(self, choices: List[str]) -> List[dict]:
        """
        Return a flat list of related rows for all choices, tagging each row
        with its source choice (the center node).
        Assumes each per-choice row includes keys like:
          - 'neighbour' (node id), 'rel_type', 'direction', 'kind' (optional).
        """
        # Normalize to list to be defensive
        if isinstance(choices, str):
            choices = [choices]

        kg = st.session_state.get("kg_result", self)
        result: List[dict] = []

        for choice in choices:
            try:
                rows = kg.fetch_related(choice)  # subclass-provided
            except Exception:
                rows = []
            if not rows:
                continue
            for r in rows:
                row = dict(r)
                row["source"] = choice  # crucial for graph & overlap logic
                # Be permissive about alternate spelling
                if "neighbour" not in row and "neighbor" in row:
                    row["neighbour"] = row.pop("neighbor")
                result.append(row)

        return result

    # ------------------------------------------------------------------ #
    # Kind & styling helpers (fixed fallback)                            #
    # ------------------------------------------------------------------ #
    def _node_kind_from(self, node_name: str):
        # 2) Try your class/graph accessors if available
        try:
            print(f"Trying to get kind from graph for node '{node_name}'")
            return self.get_node_attr(node_name, "kind")
        except Exception:
            pass

        # 3) Fallback: try the object you keep in session_state (if you use it)
        try:
            print(f"Trying to get kind from session_state graph for node '{node_name}'")
            kg = st.session_state.get("kg_result")
            print(f'kg : {kg}')
            print(f'hasattr : {hasattr(kg, "get_node_attr")}')
            if kg and hasattr(kg, "get_node_attr"):
                print(f'kind : {kg.get_node_attr(node_name, "kind")}')
                return kg.get_node_attr(node_name, "kind")
        except Exception:
            pass

        return None  # no kind known

   

    def _canonical_kind_key(self, kind) -> Optional[str]:
        """ Map raw kind (any case/spelling) to a CATEGORY_COLOR key or None if unknown."""
        if kind is None:
            return None
        ks = str(kind).strip().lower()
        for key in self.CATEGORY_COLOR.keys():
            if key == "_default":
                continue
            if key.lower() == ks:
                return key
        return None

    """
    These are the helpers to truncate. 
    """
    def _truncate(self, text: str, max_len: int = 36) -> str:
        if text is None:
            return ""
        s = str(text)
        return s if len(s) <= max_len else s[: max_len - 1] + "…"

    def _short_label(self, node_id: str) -> str:
        """
        Short, wrapped label for nodes.
        Uses a tighter limit for Papers so titles don’t blow up the layout.
        """
        name = self._get_attr(node_id, "name") or str(node_id)
        kind = self._canonical_kind_key(self._node_kind_from(node_id)) or ""
        limit = 32 if kind == "Paper" else 28
        return wrap_label(self._truncate(name, limit))

    def _node_title_json(self, node_id: str) -> str:
        """
        Metadata shown in the tooltip (like your full-graph view).
        """
        try:
            attrs = dict(self.graph.nodes[node_id])
        except Exception:
            attrs = {}
        # Ensure 'kind' is present for clarity
        attrs.setdefault("kind", self._node_kind_from(node_id))
        # Put a human name up top if missing
        attrs.setdefault("name", self._get_attr(node_id, "name") or str(node_id))
        return json.dumps(attrs, indent=2, ensure_ascii=False)

    def _color_for_kind(self, kind: Optional[str]) -> str:
        """Return the hex color for a given node kind."""
        if not kind:
            return self.CATEGORY_COLOR.get("_default", "#CCCCCC")
        k = str(kind).strip().title()
        return self.CATEGORY_COLOR.get(k, self.CATEGORY_COLOR.get("_default", "#CCCCCC"))

    def _node_style(self, *, kind, is_central: bool = False) -> dict:
        c = self._color_for_kind(kind)
        return {
            "shape": "dot",
            "size": 26 if is_central else 18,
            "font": {"color": "#ffffff", "size": 16 if is_central else 14},
            "color": {"background": c, "border": c},
            "borderWidth": 2 if is_central else 1,
        }

    def _node_kind(self, node: str) -> str:
        try:
            return self.get_node_attr(node, "kind")
        except Exception:
            return None

    def _legend_html(self, kinds) -> str:
        uniq = sorted({k for k in kinds if k})
        if not uniq:
            return ""
        chips = []
        for k in uniq:
            chips.append(
                f'<span style="display:inline-flex;align-items:center;gap:8px;'
                f'padding:4px 8px;border:1px solid #e5e7eb;border-radius:9999px;'
                f'margin:4px 6px 0 0;font:13px/1.2 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;">'
                f'<span style="width:12px;height:12px;border-radius:50%;background:{self._color_for_kind(k)};"></span>{k}'
                f'</span>'
            )
        return '<div style="margin:8px 0 4px 0;">' + "".join(chips) + "</div>"

    # ------------------------------------------------------------------ #
    # Multi-choice related fetching & visualization                      #
    # ------------------------------------------------------------------ #
    def fetch_related_list(self, choices: List[str]) -> List[dict]:
        """
        Return a flat list of related rows for all choices, tagging each row
        with its source choice (the center node).
        Assumes each per-choice row includes keys like:
          - 'neighbour' (node id), 'rel_type', 'direction', 'kind' (optional).
        """
        # Normalize to list to be defensive
        if isinstance(choices, str):
            choices = [choices]

        kg = st.session_state.get("kg_result", self)
        result: List[dict] = []

        for choice in choices:
            try:
                rows = kg.fetch_related(choice)  # subclass-provided
            except Exception:
                rows = []
            if not rows:
                continue
            for r in rows:
                row = dict(r)
                row["source"] = choice  # crucial for graph & overlap logic
                # Be permissive about alternate spelling
                if "neighbour" not in row and "neighbor" in row:
                    row["neighbour"] = row.pop("neighbor")
                result.append(row)

        return result

    def show_subgraph_streamlit(self, choices: list[str], option: str = "OR"):
        """
        Visualize neighbours for multiple choices with OR/AND semantics, 
        using category-based colors for nodes and edges.
        """
        import pandas as pd
        import streamlit as st
        from pyvis.network import Network
        
        # -------- Normalize inputs & compute mode --------
        if isinstance(choices, str):
            choices = [choices]
        option_upper = (option or "OR").strip().upper()
        if option_upper not in {"OR", "AND"}:
            option_upper = "OR"


        # -------- Collect raw rows (with 'source' attached) --------
        data = self.fetch_related_list(choices)
        temp_display = choices[0] if len(choices) == 1 else ", ".join(choices)
        st.subheader(f"Neighbours of **{temp_display}** (mode: {option_upper})")

        if not data:
            st.info("No neighbours found for the selected items.")
            return

        df = pd.DataFrame(data)
        if "source" not in df.columns:
            df["source"] = None
        if "neighbour" not in df.columns:
            st.warning("Related data is missing 'neighbour' identifiers.")
            st.dataframe(df, use_container_width=True)
            return

        # -------- Overlap logic (k = 1 for OR, k = len(choices) for AND) --------
        k = 1 if option_upper == "OR" else len(choices)
        counts = df.groupby("neighbour")["source"].nunique().reset_index(name="source_count")
        allowed_neighbours = set(counts.loc[counts["source_count"] >= k, "neighbour"].tolist())
        df_filtered = df[df["neighbour"].isin(allowed_neighbours)].copy()

        # -------- Display table --------
        df_filtered["neighbour_label"] = df_filtered["neighbour"].apply(lambda nid: self._display_label(nid))
        if "kind" not in df_filtered.columns:
            df_filtered["kind"] = None

        def _set_agg(vals):
            uniq = sorted({v for v in vals if pd.notna(v)})
            return ", ".join(uniq) if uniq else ""

        rel_agg = ("rel_type", _set_agg) if "rel_type" in df_filtered.columns else ("source", lambda s: "")
        dir_agg = ("direction", _set_agg) if "direction" in df_filtered.columns else ("source", lambda s: "")

        table = (
            df_filtered.groupby("neighbour")
            .agg(neighbour_label=("neighbour_label", "first"), rel_types=rel_agg, directions=dir_agg)
            .reset_index()
            .rename(columns={"neighbour": "neighbour_id"})
        )
        display_cols = ["neighbour_label", "rel_types", "directions"]
        table = table[[c for c in display_cols if c in table.columns]]
        table.index = range(1, len(table) + 1)
        table.index.name = "Serial No."
        st.dataframe(table, use_container_width=True)

        # -------- Visualize Graph --------
        if not st.checkbox("Visualise neighbours (PyVis)"):
            return


        net = Network(height="600px", width="100%", directed=True,
                    bgcolor="#FAFAFA", font_color="#202124")
        
        kinds_for_legend = set()

        # -------- Add center nodes --------
        for c in choices:
            knd = self._node_kind_from(c)
            kinds_for_legend.add(knd)
            net.add_node(
                c,
                label=self._display_label(c),
                title=self._node_title_json(c),
                **self.style_for(c, is_center=True)
            )

        # -------- Add neighbour nodes --------
        neighbour_ids = sorted(allowed_neighbours)
        for nid in neighbour_ids:
            knd = self._node_kind_from(nid)
            kinds_for_legend.add(knd)
            if nid not in net.node_map:
                net.add_node(
                    nid,
                    label=self._short_label(nid),
                    title=self._node_title_json(nid),
                    **self.style_for(nid)
                )

        # -------- Add colored edges --------
        edges_seen = set()
        for _, row in df_filtered.iterrows():
            src_choice = row.get("source")
            neigh = row.get("neighbour")
            rel = row.get("rel_type", "")
            direction = row.get("direction", "out")

            if not src_choice or not neigh:
                continue
            if direction == "out":
                u, v = src_choice, neigh
            else:
                u, v = neigh, src_choice

            key = (u, v, rel)
            if key in edges_seen:
                continue
            edges_seen.add(key)

            src_kind = self._node_kind_from(src_choice)
            edge_color = self._color_for_kind(src_kind)

            net.add_edge(u, v, label=rel, title=rel, color=edge_color)

        # -------- Physics & Legend --------
        net.repulsion(
            node_distance=180, central_gravity=0.25,
            spring_length=200, spring_strength=0.05, damping=0.09
        )

        legend_html = self._legend_html(kinds_for_legend)
        if legend_html:
            st.markdown(legend_html, unsafe_allow_html=True)

        st.components.v1.html(net.generate_html(), height=800, scrolling=True)

        # Optional debug
        if st.checkbox("Show resolved kinds (debug)"):
            resolved = {c: self._node_kind_from(c) for c in choices}
            for nid in neighbour_ids:
                resolved[nid] = self._node_kind_from(nid)
            st.write(resolved)


   

    #def _style_for(self, node_name: str, row: Optional[dict] = None, is_center: bool = False) -> dict:
    #    kind = self._node_kind_from(node_name, row=row)
    ##    hex_color = self._color_for_kind(kind)  # fixed fallback to _default
    #    return dict(
    #        shape="circle",
    #        font={"color": "#ffffff", "size": 16 if is_center else 14},
    #        color={"background": hex_color, "border": hex_color},
    #        borderWidth=2
    #    )
    




    # ------------------------------------------------------------------ #
    # Abstract API every concrete graph builder must implement           #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_graph(self) -> None:
        """Convert ``self.ontology_extractions`` into a graph structure.

        Concrete subclasses (e.g. `Neo4jKnowledgeGraph`) should implement the
        entire graph-creation workflow here.
        """
        pass

    @abstractmethod
    def show_graph(self):
        pass

    @abstractmethod
    def get_attr_keys(self):
        """Return all attribute names that appear on any node
        (e.g. ['layer', 'keyword', 'domain', ...])."""

    @abstractmethod
    def get_attr_values(self, key: str):
        """Return distinct values for *key* (e.g. all layer names)."""

    @abstractmethod
    def subgraph_for_attr(self, key: str, value: str):
        """Return sub-graph containing only nodes with attr[key]==value."""

    @abstractmethod
    def neighbour_subgraph(self, node_id: str):
        """Return node + first-degree neighbours (for drill-down)."""
