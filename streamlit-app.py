"""streamlit_app.py – SOKEGraph end‑to‑end Streamlit UI

🚀  New features (2025‑07‑07)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Step‑by‑step status tracker** – each phase of the pipeline is wrapped in
   `st.status`, so the UI tells you exactly where it is (retrieving → ontology
   update → ranking → graph build).
2. **Instant previews + downloads**
   • Retrieved and ranked papers appear as interactive tables.
   • CSV **and** Excel buttons let users grab the data.
   • The updated ontology file gets its own download link.
3. **Knowledge‑graph viewer**
   • For *networkx* mode we render an interactive PyVis visual directly inside
     Streamlit.
   • For *Neo4j* mode the graph is written to your Neo4j instance; you can then
     use the built‑in **Graph Query panel** below to explore a node’s neighbours
     and view them in PyVis.
4. Keeps all the upload widgets, output‑directory handling and query UI you
   already had.

💡  Replace the stub functions (`retrieve_papers_*`, `update_ontology`,
    `rank_papers`, `build_knowledge_graph`) with the real calls from your
    *sokegraph* backend.  The Streamlit logic will remain unchanged.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import List

import networkx as nx
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network

from sokegraph.sources.base_paper_source import BasePaperSource
from sokegraph.sources.semantic_scholar_source import SemanticScholarPaperSource
from sokegraph.sources.pdf_paper_source import PDFPaperSource
from sokegraph.ranking.paper_ranker import PaperRanker
from sokegraph.graph.knowledge_graph import KnowledgeGraph
from sokegraph.agents.ai_agent import AIAgent
from sokegraph.agents.openai_agent import OpenAIAgent
from sokegraph.agents.gemini_agent import GeminiAgent
from sokegraph.ontology.ontology_updater import OntologyUpdater
from sokegraph.graph.neo4j_knowledge_graph import Neo4jKnowledgeGraph
from sokegraph.agents.llama_agent import LlamaAgent
from sokegraph.agents.ollama_agent import OllamaAgent
from sokegraph.agents.claude_agent import ClaudeAgent
from sokegraph.sources.journal_api_source import JournalApiPaperSource

from sokegraph.graph.networkx_knowledge_graph import NetworkXKnowledgeGraph
import uuid

import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile, pathlib

# ─────────────────────────────────────────────────────────────────────────────
# ⚠️  Backend stubs – wire these up to SOKEGraph modules
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_papers_semantic_scholar(paper_query_file: str, number_papers: int, output_dir:str) -> pd.DataFrame:  # noqa: D401
    """Stub – replace with SemanticScholarPaperSource.

    Parameters
    ----------
    query_txt : str
        Path to a plain‑text file containing the search query/keywords.
    n : int
        Number of papers to fetch.
    """

    
    paper_source: BasePaperSource
    paper_source = SemanticScholarPaperSource(
        num_papers=int(number_papers),
        query_file=paper_query_file,
        output_dir=output_dir
    )
    
    papers = paper_source.fetch_papers()

    return papers

def retrieve_papers_journal_API(paper_query_file: str, number_papers: int, output_dir: str, api_key_file: str) -> pd.DataFrame:
    """Retrieve papers from the Web of Science Journal API.

    Parameters
    ----------
    paper_query_file : str
        Path to a plain‑text file containing the search query/keywords.
    number_papers : int
        Number of papers (journals) to fetch.
    output_dir : str
        Directory to save the output Excel file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the fetched journal metadata.
    """
    
    paper_source = JournalApiPaperSource(
        num_papers=int(number_papers),
        query_file=paper_query_file,
        output_dir=output_dir,
        api_key_file=api_key_file
    )
    
    papers = paper_source.fetch_papers()
    return papers



def retrieve_papers_from_zip(zip_path: str, output_dir: str) -> pd.DataFrame:
    """Stub – extract metadata from the PDFs in *zip_path*."""
    paper_source: BasePaperSource
    paper_source = PDFPaperSource(
                   zip_path=zip_path,
                   output_dir=output_dir
                )
    papers = paper_source.fetch_papers()
    return papers


def update_ontology(
    papers: pd.DataFrame,
    base_ontology: str,
    ai_tool: str,
    output_dir: str
) -> str:
    """Stub – update ontology with new papers and return path to updated file."""

    ontology_updater = OntologyUpdater(base_ontology, papers, ai_tool, output_dir)  # or however you instantiate it
    ontology_extractions = ontology_updater.enrich_with_papers()
    return ontology_updater.output_path

def rank_papers(
    papers,
    ontology_path: str,
    keyword_file: str,
    ai_tool,
    output_dir,
) -> pd.DataFrame:
    """Stub – score papers and return ranked DataFrame."""
    ranker = PaperRanker(ai_tool, papers, ontology_path, keyword_file, output_dir)
    output_path_files = ranker.rank_papers()
    return output_path_files


def build_knowledge_graph(
    ontology_path: str,
    kg_type: str,
    creds_path: str,
):
    """Stub – return either a NetworkX graph or write to Neo4j and return the driver."""
    print(creds_path)
    graph_builder: KnowledgeGraph
    if(kg_type == "neo4j"):
        with open(creds_path, "r") as f:
            credentials = json.load(f)
        graph_builder = Neo4jKnowledgeGraph(ontology_path, 
                                            credentials["neo4j_uri"],
                                            credentials["neo4j_user"],
                                            credentials["neo4j_pass"])
        graph_builder.build_graph()
        #driver = GraphDatabase.driver(credentials["neo4j_uri"], auth=(credentials["neo4j_user"], credentials["neo4j_pass"]))
        #return driver
        return graph_builder
    elif(kg_type == "networkx"):
        graph_builder = NetworkXKnowledgeGraph(ontology_path)
        graph_builder.build_graph()
        return graph_builder

# ─────────────────────────────────────────────────────────────────────────────
# 🖼️  UI helpers
# ─────────────────────────────────────────────────────────────────────────────

import urllib.parse
import base64

def _make_download_link(label: str, data_bytes: bytes, file_name: str, mime: str):
    """Embed *data_bytes* in a Base64 data‑URI and show a clickable link.

    Using a link avoids the Streamlit rerun that happens with
    `st.download_button`, so the rest of the app state remains visible.
    """
    b64 = base64.b64encode(data_bytes).decode()
    data_uri = f"data:{mime};base64,{urllib.parse.quote(b64)}"
    st.markdown(
        f'<a href="{data_uri}" download="{file_name}">📥 {label}</a>',
        unsafe_allow_html=True,
    )



def _show_dataframe(df: pd.DataFrame, caption: str):
    st.subheader(caption)
    st.dataframe(df, use_container_width=True)


def _display_graph(G: nx.Graph):
    net = Network(height="600px", width="100%", directed=True)
    net.barnes_hut()

    # --- Add nodes with labels ---
    for node, attrs in G.nodes(data=True):
        label = attrs.get("label", str(node))  # fallback to node ID
        net.add_node(
            n_id=node,
            label=label,
            title=json.dumps(attrs, indent=2),  # shows full metadata on hover
            color="#1f78b4"
        )

    # --- Add edges with optional labels ---
    for source, target, attrs in G.edges(data=True):
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

# ---------------------------------------------------------------------------
CATEGORY_COLOR = {
    "Paper": "#FF5733",
    "Author": "#1f77b4",
    "Keyword": "#2ca02c",
    "Institution": "#9467bd",
    "Journal": "#d62728",
    "_default": "#97C2FC",
}


def wrap_label(text: str, max_width: int = 15) -> str:
    """Insert line breaks to wrap long labels inside a PyVis node."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current + " " + word) <= max_width:
            current = (current + " " + word).strip()
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


# Neo4j helpers --------------------------------------------------------------

def fetch_node_names(driver) -> List[str]:
    query = """
    MATCH (n) WHERE n.name IS NOT NULL RETURN DISTINCT n.name AS name ORDER BY name
    """
    with driver.session() as sess:
        return sess.run(query).value()



def _get_driver(creds_path: Path):
    creds = json.loads(Path(creds_path).read_text())
    return GraphDatabase.driver(creds["neo4j_uri"], auth=(creds["neo4j_user"], creds["neo4j_pass"]))

# ---------------------------------------------------------------------------

def _needs_api_key(agent: str) -> bool:
    return agent.lower() in {"openai", "gemini", "llama", "claude", "deepseek",  "Journal API"}


def _needs_kg_cred(kg_type: str) -> bool:
    return kg_type.lower() in {"neo4j"}


def _save_upload(uploaded_file, suffix: str) -> Path | None:
    if uploaded_file is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    return Path(tmp.name)

# ════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="SOKEGraph", layout="wide")
    st.title("📚 SOKEGraph – Paper Retrieval → Ontology → Ranking → KG")

    left, right = st.columns([1.2, 2])

    # ---------------- LEFT COLUMN: paper source ---------------------------
    with left:
        st.subheader("Paper Source")
        paper_mode = st.radio("", ["Semantic Scholar", "Journal API", "PDF ZIP"], key="src")

    # ---------------- RIGHT COLUMN: configuration -------------------------
    with right:
        st.subheader("⚙️ Configuration")

        if paper_mode == "Semantic Scholar":
            num_papers = st.number_input("Number of papers", 1, 200, 10) #should change upper bound
            query_file = st.file_uploader("Upload query file (.txt)", type=["txt"])
            api_key_for_journal_api_file = None
            pdf_zip = None
        elif paper_mode == "Journal API":
            num_papers = st.number_input("Number of papers", 1, 200, 10) #should change upper bound
            query_file = st.file_uploader("Upload query file (.txt)", type=["txt"])
            api_key_for_journal_api_file = st.file_uploader("Upload API key for journal API (.txt)", type=["txt"]) 
            pdf_zip = None
        else:
            pdf_zip = st.file_uploader("Upload PDF ZIP", type=["zip"])
            api_key_for_journal_api_file = None
            query_file = None
            num_papers = None

        ontology_file = st.file_uploader("Upload base ontology file (.jsonl)", type=["json"])
        agent = st.selectbox("AI Agent", ["openAI", "gemini", "llama", "ollama", "claude"])
        api_key_file = st.file_uploader("Upload API key (.txt)", type=["txt"]) if _needs_api_key(agent) else None
        keywords_file = st.file_uploader("Upload keywords file (.txt)", type=["txt"])
        kg_type = st.selectbox("Knowledge Graph", ["neo4j", "networkx"])
        kg_credentials = st.file_uploader("Upload KG credentials (.json)", type=["json"]) if _needs_kg_cred(kg_type) else None

        output_dir = Path("external/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        run_btn = st.button("🚀 Run Pipeline")




    # ---------------------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------------------
    if run_btn:
        # 🔍 Validate required inputs
        missing_inputs = []

        if paper_mode in {"Semantic Scholar", "Journal API"}:
            if not query_file:
                missing_inputs.append("Query file (.txt)")
            if not num_papers:
                missing_inputs.append("Number of papers")
            elif num_papers <= 0:
                missing_inputs.append("Number of papers must be greater than 0")

        if paper_mode == "Journal API" and not api_key_for_journal_api_file:
            missing_inputs.append("Journal API key file (.txt)")

        if paper_mode == "PDF ZIP" and not pdf_zip:
            missing_inputs.append("PDF ZIP file")

        if not ontology_file:
            missing_inputs.append("Base ontology file (.jsonl)")

        if _needs_api_key(agent) and not api_key_file:
            missing_inputs.append(f"{agent} API key (.txt)")

        if not keywords_file:
            missing_inputs.append("Keywords file (.txt)")

        if _needs_kg_cred(kg_type) and not kg_credentials:
            missing_inputs.append("KG credentials (.json)")

        if missing_inputs:
            st.error("❌ Please upload or select the following before running:\n\n" + "\n".join(f"• {item}" for item in missing_inputs))
            st.stop()
        # (validation code unchanged) …

        # Save uploads
        api_key_path = _save_upload(api_key_file, ".txt") if api_key_file else None
        query_path   = _save_upload(query_file, ".txt")   if query_file   else None
        pdf_path     = _save_upload(pdf_zip,  ".zip")     if pdf_zip      else None
        keywords_path= _save_upload(keywords_file, ".txt")
        ontology_path=_save_upload(ontology_file, ontology_file.name.split(".")[-1])
        creds_path   = _save_upload(kg_credentials, ".json") if kg_credentials else None

        # 0. AI agent (unchanged) …
        ai_tool: AIAgent = {
            "openAI":  OpenAIAgent,
            "gemini":  GeminiAgent,
            "llama":   LlamaAgent,
            "ollama":  OllamaAgent,
            "claude":  ClaudeAgent,
        }[agent](api_key_path)

        # 🏃 Run pipeline
        with st.status("Running pipeline…", expanded=True) as status:
            # 1️⃣ Retrieve papers
            status.write("1/4 • Retrieving papers…")
            if paper_mode == "Semantic Scholar":
                papers = retrieve_papers_semantic_scholar(str(query_path), int(num_papers), output_dir)

            elif paper_mode == "PDF Zip":
                papers = retrieve_papers_from_zip(str(pdf_path), output_dir)

            elif paper_mode == "Journal API":
                papers = retrieve_papers_journal_API(str(query_path), int(num_papers), output_dir, api_key_for_journal_api_file)

            else:
                raise ValueError(f"Unsupported paper source mode: {paper_mode}")
            papers_df = pd.DataFrame(papers)
            status.update(label="✅ Papers retrieved")

            # 2️⃣ Update ontology
            status.update(label="2/4 • Updating ontology…")
            updated_ontology_path = update_ontology(papers, str(ontology_path), ai_tool, output_dir)
            status.update(label="✅ Ontology updated")

            # 3️⃣ Rank papers
            status.update(label="3/4 • Ranking papers…")
            ranked_path_files = rank_papers(papers, str(updated_ontology_path), str(keywords_path), ai_tool, output_dir)
            print(f"print ranked path : {papers}")
            ranked_df_parts = [pd.read_csv(p) for p in ranked_path_files.values()]
            status.update(label="✅ Papers ranked")

            # 4️⃣ Knowledge graph
            status.update(label="4/4 • Building knowledge graph…")
            kg_result = build_knowledge_graph(str(updated_ontology_path), kg_type, str(creds_path) if creds_path else None)
            status.update(label="✅ Knowledge graph built")

        # ✨  KEEP everything in session_state so it re‑renders on every rerun
        st.session_state["papers_df"]      = papers_df
        st.session_state["ontology_path"]  = updated_ontology_path
        st.session_state["ranked_parts"]   = ranked_df_parts
        st.session_state["kg_result"]      = kg_result
        st.session_state["kg_type"]        = kg_type
    
    # ---------------------------------------------------------------------
    # ALWAYS SHOW CACHED OUTPUTS
    # ---------------------------------------------------------------------
    if "papers_df" in st.session_state:
        _show_dataframe(st.session_state["papers_df"], "Retrieved Papers")
        _make_download_link("Download papers CSV",
                            st.session_state["papers_df"].to_csv(index=False).encode(),
                            "papers.csv", "text/csv")

    if "ontology_path" in st.session_state:
        st.subheader("Updated Ontology")
        _make_download_link("Download ontology",
                            Path(st.session_state["ontology_path"]).read_bytes(),
                            Path(st.session_state["ontology_path"]).name,
                            "application/rdf+xml")

    if "ranked_parts" in st.session_state:
        for df in st.session_state["ranked_parts"]:
            bucket = df.iloc[0].get("rank_bucket", "Ranked")
            _show_dataframe(df, f"{bucket.title()}‑confidence Papers")

    if "kg_result" in st.session_state:
        st.markdown("---")
        st.subheader("Knowledge Graph (full)")
        if st.session_state["kg_type"] == "networkx":
            html = st.session_state["kg_result"].show_graph()
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            st.session_state["kg_result"].show_graph()

        # ---- Node‑only explorer ----
        display_node_only_graph(st.session_state["kg_result"])
        
        #from sokegraph.interactive_graph_view import InteractiveGraphView
        #InteractiveGraphView(st.session_state["kg_result"]).render()
        st.subheader("Interactive Knowledge‑Graph")

        # ── Select node type (label)
        node_labels = st.session_state["kg_result"].get_node_labels()
        selected_label = st.selectbox("Select node type", node_labels)

        # ── Select value (e.g., name of a Layer/Keyword/etc.)
        print(type(selected_label))
        values = st.session_state["kg_result"].get_attr_values(selected_label)
        print(values)
        choice = st.selectbox("Value", values, key="attr_value")
        
        #driver = st.session_state.neo4j_driver

        if choice:
            data = st.session_state["kg_result"].fetch_related(choice)
            st.subheader(f"Neighbours of **{choice}**")
            st.dataframe(data, use_container_width=True)

            if st.checkbox("Visualise neighbours (PyVis)"):
                net = Network(height="500px", width="100%", directed=True)
                style = dict(shape="circle", font={"color": "#ffffff", "size": 14}, color={"background": "#FF5733", "border": "#FF5733"}, borderWidth=2)
                net.add_node(choice, label=wrap_label(choice), **style)

                for row in data:
                    n = row["neighbour"]
                    rel = row["rel_type"]
                    if n not in net.node_map:
                        net.add_node(n, label=wrap_label(n), **style)
                    if row["direction"] == "out":
                        net.add_edge(choice, n, label=rel)
                    else:
                        net.add_edge(n, choice, label=rel)
                net.repulsion(node_distance=180, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
                html = net.generate_html()
                st.components.v1.html(html, height=800, scrolling=True)
        



    # ─────────────────────────────────────────────────────────────────────
    # Neo4j Graph Query panel
    # ─────────────────────────────────────────────────────────────────────
    # if "neo4j_driver" in st.session_state and st.session_state.neo4j_driver:
    #     st.markdown("---")
    #     st.title("SOKEGraph – explore neighbours")
    #     driver = st.session_state.neo4j_driver

    #     names = fetch_node_names(driver)
    #     choice = st.selectbox("Select a node", names, index=None, placeholder="Choose…")

    #     if choice:
    #         data = fetch_related(driver, choice)
    #         st.subheader(f"Neighbours of **{choice}**")
    #         st.dataframe(data, use_container_width=True)

    #         if st.checkbox("Visualise neighbours (PyVis)"):
    #             net = Network(height="500px", width="100%", directed=True)
    #             style = dict(shape="circle", font={"color": "#ffffff", "size": 14}, color={"background": "#FF5733", "border": "#FF5733"}, borderWidth=2)
    #             net.add_node(choice, label=wrap_label(choice), **style)

    #             for row in data:
    #                 n = row["neighbour"]
    #                 rel = row["rel_type"]
    #                 if n not in net.node_map:
    #                     net.add_node(n, label=wrap_label(n), **style)
    #                 if row["direction"] == "out":
    #                     net.add_edge(choice, n, label=rel)
    #                 else:
    #                     net.add_edge(n, choice, label=rel)
    #             net.repulsion(node_distance=180, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
    #             html = net.generate_html()
    #             st.components.v1.html(html, height=800, scrolling=True)

    #     driver.close()


def display_node_only_graph(kg_result):
    """
    Add a section to show only nodes based on selected type,
    without clearing or hiding other Streamlit content.
    """
    node_labels = st.session_state["kg_result"].get_node_labels()
    with st.expander("🔍 Explore Nodes by Type", expanded=True):
        view_mode = st.radio(
            "Nodes to display:",
            #["All", "Layer", "Category", "Keyword", "Paper", "MetaData"],
            node_labels,
            horizontal=True,
            key="node_display_mode"
        )

        if hasattr(kg_result, "generate_nodes_html"):
            html = kg_result.generate_nodes_html(view_mode)
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            st.warning("⚠️ The current knowledge graph object doesn't support `generate_nodes_html`.")





if __name__ == "__main__":
    main()
