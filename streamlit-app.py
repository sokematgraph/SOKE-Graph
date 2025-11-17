from __future__ import annotations
import streamlit as st



import json
import tempfile
from pathlib import Path
from typing import List

import networkx as nx
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network

from sokegraph.sources.base_paper_source import BasePaperSource
from sokegraph.sources.semantic_scholar_source import SemanticScholarPaperSource
from sokegraph.sources.pdf_paper_source import PDFPaperSource

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
from sokegraph.utils.functions import load_papers
from sokegraph.ranking.paper_ranker import PaperRanker
from sokegraph.ui.streamlit_helper import show_json_file
from sokegraph.util.logger import LOG
import uuid

import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile
import os
import torch

import time 
from datetime import datetime

from sokegraph.ui.streamlit_helper import _make_download_link, _show_dataframe_papers, show_ranked_results_with_links




def apply_theme_styles():
    """Apply Streamlit theme colors dynamically to custom tables and graphs."""
    theme = st.get_option("theme.base")
    primary = st.get_option("theme.primaryColor")
    bg = st.get_option("theme.backgroundColor")
    secondary = st.get_option("theme.secondaryBackgroundColor")
    text = st.get_option("theme.textColor")

    # --- universal CSS based on theme ---
    st.markdown(f"""
    <style>
    /* App background */
    html, body, .stApp {{
        background-color: {bg} !important;
        color: {text} !important;
        font-family: "Segoe UI", sans-serif;
    }}

    /* DataFrame styling */
    .stDataFrame table {{
        background-color: {secondary} !important;
        color: {text} !important;
        border: 1px solid rgba(120,120,120,0.3);
        border-radius: 5px;
    }}
    .stDataFrame th {{
        background-color: {primary}22 !important;   /* light version of primary */
        color: {text} !important;
        font-weight: 600;
    }}
    .stDataFrame tr:hover td {{
        background-color: {primary}15 !important;
    }}

    /* Links inside tables */
    .stDataFrame a {{
        color: {primary} !important;
        text-decoration: none;
        font-weight: 500;
    }}
    .stDataFrame a:hover {{
        text-decoration: underline;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Return theme colors for your graphs too
    return dict(primary=primary, bg=bg, secondary=secondary, text=text)


# ─────────────────────────────────────────────────────────────
# ⚡ DEVICE SELECTION (GPU / CPU / MPS)
# ─────────────────────────────────────────────────────────────
st.sidebar.subheader("⚡ Compute Device")
force_cpu = st.sidebar.checkbox("Force CPU (ignore GPU)", value=False)

try:
    if not force_cpu and torch.cuda.is_available():
        device = "cuda"
    elif not force_cpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
except ImportError:
    device = "cpu"

os.environ["SOKEGRAPH_DEVICE"] = device
st.sidebar.write(f"Using device: `{device}`")

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
    return ontology_updater.enrich_with_papers()
    

def rank_papers(
    papers_path,
    ontology_path: str,
    keyword_file: str,
    ai_tool,
    output_dir,
) -> pd.DataFrame:
    """
    Score papers with static, dynamic, and HRM (if weights provided).
    """
    ranker = PaperRanker(
        ai_tool,
        papers_path,
        ontology_path,
        keyword_file,
        output_dir,
        hrm_use_neural=False,  # Disable neural HRM to use fallback
        hrm_use_simple_fallback=False,  # Use simple fallback from paper_ranker.py.txt
        polarity_lambda=0.5,  # Simple fallback penalty parameter
        bypass_filtering=True
    )
    output_csv_paths, output_paths_all = ranker.rank_papers()
    return output_csv_paths, output_paths_all


def build_knowledge_graph(
    ontology_path: str,
    papers_path: str,
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
                                            papers_path, 
                                            credentials["neo4j_uri"],
                                            credentials["neo4j_user"],
                                            credentials["neo4j_pass"])
        graph_builder.build_graph()
        #driver = GraphDatabase.driver(credentials["neo4j_uri"], auth=(credentials["neo4j_user"], credentials["neo4j_pass"]))
        #return driver
        return graph_builder
    elif(kg_type == "networkx"):
        graph_builder = NetworkXKnowledgeGraph(ontology_path, papers_path)
        graph_builder.build_graph()
        return graph_builder

# ─────────────────────────────────────────────────────────────────────────────
# 🖼️  UI helpers
# ─────────────────────────────────────────────────────────────────────────────

import urllib.parse
import base64

import mimetypes
from pathlib import Path

def show_download_links(output_paths_all: dict):
    """
    Display download links for all files in output_paths_all using _make_download_link.

    Parameters:
    - output_paths_all: dict
        Example:
        {
            "high": {"csv": "high.csv", "xlsx": "high.xlsx"},
            "low": {"csv": "low.csv", "xlsx": "low.xlsx"},
            "unknown": {"csv": "unknown.csv", "xlsx": "unknown.xlsx"}
        }
    """
    if not output_paths_all:
        st.warning("No files to download.")
        return

    st.subheader("Download Results")

    for level, files_dict in output_paths_all.items():
        st.markdown(f"**{level.capitalize()} relevance papers:**")
        for file_type, file_path_str in files_dict.items():
            file_path = Path(file_path_str)
            if file_path.exists():
                # Read file as bytes
                with open(file_path, "rb") as f:
                    data_bytes = f.read()
                # Determine MIME type
                mime = "text/csv" if file_path.suffix.lower() == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                # Create download link
                _make_download_link(file_path.name, data_bytes, file_path.name, mime)
            else:
                st.warning(f"File not found: {file_path}")

import mimetypes
from pathlib import Path

def show_download_links(output_paths_all: dict):
    """
    Display download links for all files in output_paths_all using _make_download_link.

    Parameters:
    - output_paths_all: dict
        Example:
        {
            "high": {"csv": "high.csv", "xlsx": "high.xlsx"},
            "low": {"csv": "low.csv", "xlsx": "low.xlsx"},
            "unknown": {"csv": "unknown.csv", "xlsx": "unknown.xlsx"}
        }
    """
    if not output_paths_all:
        st.warning("No files to download.")
        return

    st.subheader("Download Results")

    for level, files_dict in output_paths_all.items():
        st.markdown(f"**{level.capitalize()} relevance papers:**")
        for file_type, file_path_str in files_dict.items():
            file_path = Path(file_path_str)
            if file_path.exists():
                # Read file as bytes
                with open(file_path, "rb") as f:
                    data_bytes = f.read()
                # Determine MIME type
                mime = "text/csv" if file_path.suffix.lower() == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                # Create download link
                _make_download_link(file_path.name, data_bytes, file_path.name, mime)
            else:
                st.warning(f"File not found: {file_path}")

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

# ---------------------------------------------------------------------------
CATEGORY_COLOR = {
    "Paper": "#FF5733",
    "Author": "#1f77b4",
    "Keyword": "#2ca02c",
    "Institution": "#9467bd",
    "Journal": "#d62728",
    "_default": "#97C2FC",
}

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


def _read_keywords_list(txt_path: Path) -> List[str]:
    """ Extract the keywords from the user query file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="SOKEGraph", layout="wide")
    apply_theme_styles()
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
            num_papers = st.number_input("Number of papers", 1, 1000, 10) #should change upper bound
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

        ontology_file = st.file_uploader("Upload base ontology file (.json) (optional)", type=["json"])
        field_of_interest = st.text_input("Field of interest (e.g. materials science, biology, medicine)", value="materials science")
        agent = st.selectbox("AI Agent", ["openAI", "gemini", "llama", "ollama", "claude"])
        api_key_file = st.file_uploader("Upload API key (.txt)", type=["txt"]) if _needs_api_key(agent) else None
        keywords_file = st.file_uploader("Upload keywords file (.txt)", type=["txt"])
        kg_type = st.selectbox("Knowledge Graph", ["networkx", "neo4j"])
        kg_credentials = st.file_uploader("Upload KG credentials (.json)", type=["json"]) if _needs_kg_cred(kg_type) else None

        output_dir = Path("external/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        run_btn = st.button("🚀 Run Pipeline")

    # ---------------------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------------------
    if run_btn:
        # timings container 
        timings = []
        t0_all = time.perf_counter()
        t0_all_wall  = datetime.now()
        
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


        """
        We're making this upload optional - if not provided, we can use a default ontology (stored locally ATM).
        """
        # if not ontology_file:
        #     missing_inputs.append("Base ontology file (.jsonl)")

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
        api_keys_path = _save_upload(api_key_file, ".txt") if api_key_file else None
        query_path   = _save_upload(query_file, ".txt")   if query_file   else None
        pdf_path     = _save_upload(pdf_zip,  ".zip")     if pdf_zip      else None
        keywords_path= _save_upload(keywords_file, ".txt")
        # ontology_path = Path("/Users/hrishilogani/Desktop/soke-test-data/Ontology.json")
        if ontology_file:
            ontology_path=_save_upload(ontology_file, ontology_file.name.split(".")[-1])
        creds_path = _save_upload(kg_credentials, ".json") if kg_credentials else None

        # 0. AI agent (unchanged) …
        ai_tool_cls: AIAgent = {
            "openAI":  OpenAIAgent,
            "gemini":  GeminiAgent,
            "llama":   LlamaAgent,
            "ollama":  OllamaAgent,
            "claude":  ClaudeAgent,
        }[agent]

        if agent == "ollama": 
            ai_tool = ai_tool_cls(field_of_interest=field_of_interest)
        else:
            ai_tool = ai_tool_cls(api_keys_path, field_of_interest=field_of_interest)

        # 🏃 Run pipeline
        with st.status("Running pipeline…", expanded=True) as status:
            # 1️⃣ Retrieve papers)
            status.update(label="1/4 • Retrieving papers…")
            t1_start = time.perf_counter(); w1_start = datetime.now()
            if paper_mode == "Semantic Scholar":
                papers_path = retrieve_papers_semantic_scholar(str(query_path), int(num_papers), output_dir)

            elif paper_mode == "PDF Zip":
                papers_path = retrieve_papers_from_zip(str(pdf_path), output_dir)

            elif paper_mode == "Journal API":
                papers_path = retrieve_papers_journal_API(str(query_path), int(num_papers), output_dir, api_key_for_journal_api_file)

            else:
                raise ValueError(f"Unsupported paper source mode: {paper_mode}")

            
            # papers_path = 'external/output/papers_metadata.xlsx'  # Replace with actual path
            papers = load_papers(papers_path)
            papers_df = pd.DataFrame(papers)
            t1_end = time.perf_counter(); w1_end = datetime.now()
            timings.append({
                "step": "retrieve_papers",
                "start_time": w1_start.isoformat(timespec="seconds"),
                "end_time": w1_end.isoformat(timespec="seconds"),
                "duration": round(t1_end - t1_start, 3)
            })

            status.update(label="✅ Papers retrieved")

            # 2️⃣ Update ontology
            status.update(label="2/4 • Updating ontology…")
            t2_start = time.perf_counter(); w2_start = datetime.now()
            with open(query_path, "r", encoding="utf-8") as f:
                user_keywords = [ln.strip() for ln in f if ln.strip()]

            # this is updating the base ontology
            custom_prompt = None
            OntologyUpdater.enrich_base_with_keywords(
                base_ontology_path=str(ontology_path),
                keywords=user_keywords,
                ai_tool=ai_tool,
                prompt_overrides=custom_prompt,
            )
            # this is updating the base ontology
            updated_ontology_path = update_ontology(papers_path, str(ontology_path), ai_tool, output_dir)
            t2_end = time.perf_counter(); w2_end = datetime.now()
            timings.append({
                "step": "update_ontology",
                "start_time": w2_start.isoformat(timespec="seconds"),
                "end_time": w2_end.isoformat(timespec="seconds"),
                "duration": round(t2_end - t2_start, 3)
            })
            status.update(label="✅ Ontology updated")
            # updated_ontology_path = "external/output/updated_ontology.json"
            # 3️⃣ Rank papers
            status.update(label="3/4 • Ranking papers…")
            t3_start = time.perf_counter(); w3_start = datetime.now()
            output_csv_paths, output_paths_all = rank_papers(papers_path, str(updated_ontology_path), str(keywords_path), ai_tool, output_dir)
            print(f"print ranked path : {papers}")
            print(f"ranked_path_files : {output_csv_paths}")
            ranked_df_parts = [pd.read_csv(p) for p in output_csv_paths.values()]
            t3_end = time.perf_counter(); w3_end = datetime.now()
            timings.append({
                "step": "rank_papers",
                "start_time": w3_start.isoformat(timespec="seconds"),
                "end_time": w3_end.isoformat(timespec="seconds"),
                "duration": round(t3_end - t3_start, 3)
            })
            status.update(label="✅ Papers ranked")

            # 4️⃣ Knowledge graph
            status.update(label="4/4 • Building knowledge graph…")
            t4_start = time.perf_counter(); w4_start = datetime.now()
            kg_result = build_knowledge_graph(str(updated_ontology_path), papers_path, kg_type, str(creds_path) if creds_path else None)
            t4_end = time.perf_counter(); w4_end = datetime.now()
            timings.append({
                "step": "build_knowledge_graph",
                "start_time": w4_start.isoformat(timespec="seconds"),
                "end_time": w4_end.isoformat(timespec="seconds"),
                "duration": round(t4_end - t4_start, 3)
            })
            status.update(label="✅ Knowledge graph built")


            t_all_end = time.perf_counter(); w_all_end = datetime.now()
            timings.append({
                "step": "Total • End-to-end",
                "start_time": t0_all_wall.isoformat(timespec="seconds"),
                "end_time":   w_all_end.isoformat(timespec="seconds"),
                "elapsed_sec": round(t_all_end - t0_all, 3),
            })


        # ✨  KEEP everything in session_state so it re‑renders on every rerun
        st.session_state["papers_df"]      = papers_df
        st.session_state["ontology_path"]  = updated_ontology_path
        st.session_state["ranked_parts"]   = output_csv_paths
        st.session_state["kg_result"]      = kg_result
        st.session_state["kg_type"]        = kg_type
        st.session_state["output_paths_all"] = output_paths_all

            # Save timings to Excel and expose in UI
        timings_df = pd.DataFrame(timings)
        timings_df["elapsed_sec"] = timings_df["elapsed_sec"].fillna(0)
        LOG.info(f"📊 Pipeline timings: {timings_df}")
        if not timings_df.empty:
            timings_df["elapsed_hms"] = timings_df["elapsed_sec"].apply(_fmt_hms)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            timings_path = f"external/output/pipeline_timings.xlsx"
            timings_df.to_excel(timings_path, index=False)


    
    # ---------------------------------------------------------------------
    # ALWAYS SHOW CACHED OUTPUTS
    # ---------------------------------------------------------------------
    if "papers_df" in st.session_state:
        df = st.session_state["papers_df"]
        _show_dataframe_papers(df=df, header="Retrieved Papers", type_df="retrieved")
        _make_download_link(
            "Download papers CSV",
            df.to_csv(index=False).encode(),
            "papers.csv",
            "text/csv",
        )

    if "ontology_path" in st.session_state:
        st.subheader("Updated Ontology")
        show_json_file(st.session_state["ontology_path"])
        _make_download_link("Download ontology",
                            Path(st.session_state["ontology_path"]).read_bytes(),
                            Path(st.session_state["ontology_path"]).name,
                            "application/rdf+xml")

    if "ranked_parts" in st.session_state:
        for type_rank, path in st.session_state["ranked_parts"].items():
            df = pd.read_csv(path)
            _show_dataframe_papers(df=df, header=f"Ranked-Confidence Papers-{type_rank}", type_df="results")
        
        
        show_download_links(st.session_state["output_paths_all"])

    if "kg_result" in st.session_state:
        st.markdown("---")
        st.subheader("Knowledge Graph (full)")
        st.session_state["kg_result"].show_graph_streamlit()

        # ── Select node type (label)
        st.subheader("Interactive Knowledge‑Graph")
        node_labels = st.session_state["kg_result"].get_node_labels()
        selected_label = st.selectbox("Select node type", node_labels)

        # ── Select value (e.g., name of a Layer/Keyword/etc.)
        print(type(selected_label))
        values = st.session_state["kg_result"].get_attr_values(selected_label)
        print(values)

        # allow multi-select
        selected_values = st.multiselect("Select values", values, key="attr_values")

        # allow the user to select a filter type 
        filter_logic = st.radio("Filter logic", ["AND", "OR"], horizontal=True, key="filter_logic")

        if selected_values:
            st.session_state["kg_result"].show_subgraph_streamlit(selected_values, option=filter_logic)


def display_node_only_graph(kg_result):
    """
    Add a section to show only nodes based on selected type,
    without clearing or hiding other Streamlit content.
    """
    print(f"graph {kg_result}")
    node_labels = kg_result.get_node_labels()
    with st.expander("", expanded=True):
        view_mode = st.radio(
            "Nodes to display:",
            node_labels,
            horizontal=True,
            key="node_display_mode"
        )

        if hasattr(kg_result, "generate_nodes_html"):
            html = kg_result.generate_nodes_html(view_mode)
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            st.warning("⚠️ The current knowledge graph object doesn't support `generate_nodes_html`.")

# --- Private Helpers --- 

def _fmt_hms(seconds: float) -> str:
    secs = int(round(seconds))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()