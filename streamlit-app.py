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

# Fix for Windows Unicode console output issues
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import io
import json
import inspect
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
from sokegraph.utils.formatters import make_url_link, make_doi_link

from sokegraph.graph.networkx_knowledge_graph import NetworkXKnowledgeGraph
from sokegraph.utils.functions import load_papers
from sokegraph.ranking.paper_ranker import PaperRanker
from sokegraph.ui.streamlit_helper import show_json_file
from sokegraph.util.logger import LOG
import uuid

import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile, pathlib
import os
import torch

import time 
from datetime import datetime

from sokegraph.ui.streamlit_helper import _make_download_link, _show_dataframe_papers, show_ranked_results_with_links

# --- Dark/Light Theme Compatibility CSS ---

def inject_theme_css():
    # Theme detection with fallback
    try:
        theme_base = st.get_option("theme.base")
        is_dark = theme_base == "dark"
    except:
        # Fallback: assume light theme if can't detect
        is_dark = False
    
    BG  = "#0E1117" if is_dark else "#FFFFFF"
    SBG = "#262730" if is_dark else "#F0F2F6"
    TXT = "#FAFAFA" if is_dark else "#262626"
    BRD = "#444444" if is_dark else "#D3D3D3"

    st.markdown(f"""
    <style>
      /* Make sure app text follows the theme */
      .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li {{
        color: {TXT} !important;
      }}

      /* Tables generated via to_html / markdown */
      .themed table, .themed th, .themed td {{
        border: 1px solid {BRD} !important;
        border-collapse: collapse !important;
      }}
      .themed th {{
        background: {SBG} !important;
        color: {TXT} !important;
        font-weight: 600 !important;
        padding: 8px !important;
      }}
      .themed td {{
        background: {BG} !important;
        color: {TXT} !important;
        padding: 8px !important;
      }}
      .themed tr:nth-child(even) td {{
        background: rgba(255,255,255,{0.03 if is_dark else 0.6}) !important;
      }}

      /* Streamlit dataframe widget styling */
      [data-testid="stDataFrame"] {{
        background-color: {BG} !important;
      }}
      
      [data-testid="stDataFrame"] * {{
        color: {TXT} !important;
      }}
      
      /* Dataframe header cells */
      [data-testid="stDataFrame"] [role="columnheader"] {{
        background-color: {SBG} !important;
        color: {TXT} !important;
        border-color: {BRD} !important;
      }}
      
      /* Dataframe body cells */
      [data-testid="stDataFrame"] [role="gridcell"] {{
        background-color: {BG} !important;
        color: {TXT} !important;
        border-color: {BRD} !important;
      }}
      
      /* Dataframe row headers/index */
      [data-testid="stDataFrame"] [role="rowheader"] {{
        background-color: {SBG} !important;
        color: {TXT} !important;
        border-color: {BRD} !important;
      }}

      /* Code blocks & JSON boxes */
      pre, code, .themed pre, .themed code {{
        background: {SBG} !important;
        color: {TXT} !important;
      }}

      /* Links */
      a {{
        color: var(--primary-color, #4CAF50) !important;
      }}
    </style>
    """, unsafe_allow_html=True)




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

def retrieve_papers_semantic_scholar(paper_query_file: str, number_papers: int, output_dir:str) -> pd.DataFrame:
    """Retrieve papers from Semantic Scholar API.
    
    Fetches research papers using the Semantic Scholar API based on
    queries from a text file. Papers are deduplicated and enriched
    with missing abstract/venue information.

    Parameters
    ----------
    paper_query_file : str
        Path to a plain-text file containing search queries (one per line)
    number_papers : int
        Maximum number of unique papers to retrieve
    output_dir : str
        Directory where output Excel file will be saved
        
    Returns
    -------
    str
        Path to Excel file containing paper metadata
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
    """Extract paper metadata from PDFs in a ZIP file.
    
    Unzips the archive, extracts titles from PDF metadata, queries
    Semantic Scholar API for each title, and returns structured metadata.

    Parameters
    ----------
    zip_path : str
        Path to ZIP file containing PDF papers
    output_dir : str
        Directory where extracted PDFs and output will be stored
        
    Returns
    -------
    str
        Path to Excel file containing paper metadata
    """
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
    """Update ontology with keywords extracted from research papers.
    
    Uses an AI agent to extract domain-specific keywords from paper
    abstracts and merge them into the existing ontology structure.

    Parameters
    ----------
    papers : pd.DataFrame
        DataFrame containing paper metadata with abstracts
    base_ontology : str
        Path to base ontology JSON file
    ai_tool : AIAgent
        AI agent instance for keyword extraction
    output_dir : str
        Directory where updated ontology will be saved
        
    Returns
    -------
    str
        Path to the updated ontology JSON file
    """
    ontology_updater = OntologyUpdater(base_ontology, papers, ai_tool, output_dir)
    return ontology_updater.enrich_with_papers()
    

def rank_papers(
    papers_path,
    ontology_path: str,
    keyword_file: str,
    ai_tool,
    output_dir,
    layer_priority_weights: dict = None,
) -> pd.DataFrame:
    """Rank papers by relevance to user-specified keywords.
    
    Scores papers using multiple ranking methods (static, dynamic, HRM)
    based on ontology keyword matches in titles and abstracts.

    Parameters
    ----------
    papers_path : str
        Path to Excel file containing paper metadata
    ontology_path : str
        Path to ontology JSON file
    keyword_file : str
        Path to text file containing user query keywords
    ai_tool : AIAgent
        AI agent instance for query classification and expansion
    output_dir : str
        Directory where ranking results will be saved
    layer_priority_weights : dict, optional
        Dictionary mapping layer names to priority weights (default: None)
        
    Returns
    -------
    tuple[dict, dict]
        Two dictionaries: 
        - output_csv_paths: paths to CSV files with rankings
        - output_paths_all: paths to all output formats (CSV, JSON, GraphML, etc.)
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
        bypass_filtering=True,
        layer_priority_weights=layer_priority_weights  # Pass custom weights
    )
    output_csv_paths, output_paths_all = ranker.rank_papers()
    return output_csv_paths, output_paths_all


def build_knowledge_graph(
    ontology_path: str,
    papers_path: str,
    kg_type: str,
    creds_path: str,
):
    """Build a knowledge graph from ontology and paper data.
    
    Creates either a Neo4j or NetworkX knowledge graph representation
    of the ontology structure and paper connections.

    Parameters
    ----------
    ontology_path : str
        Path to updated ontology JSON file
    papers_path : str
        Path to Excel file containing paper metadata
    kg_type : str
        Type of knowledge graph: "neo4j" or "networkx"
    creds_path : str
        Path to Neo4j credentials JSON file (required if kg_type="neo4j")
        
    Returns
    -------
    KnowledgeGraph
        Knowledge graph builder instance (Neo4jKnowledgeGraph or NetworkXKnowledgeGraph)
    """
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
    st.title("📚 SOKEGraph – Paper Retrieval → Ontology → Ranking → KG")

    left, right = st.columns([1.2, 2])

    # ---------------- LEFT COLUMN: paper source ---------------------------
    with left:
        st.subheader("Paper Source")
        paper_mode = st.radio("", ["Semantic Scholar", "Journal API"], key="src")

    # ---------------- RIGHT COLUMN: configuration -------------------------
    with right:
        st.subheader("⚙️ Configuration")

        if paper_mode == "Semantic Scholar":
            num_papers = st.number_input("Number of papers", 1, 1000, 10) #should change upper bound
            api_key_for_journal_api_file = None
            
        else:
            num_papers = st.number_input("Number of papers", 1, 200, 10) #should change upper bound
            api_key_for_journal_api_file = st.file_uploader("Upload API key for journal API (.txt)", type=["txt"]) 

        query_file = st.file_uploader("Paper Query file (.txt)", type=["txt"])
        pdf_zip = None

        ontology_file = st.file_uploader("Base Ontology file (.json) (optional)", type=["json"])
        
        # ── LAYER PRIORITY WEIGHTS CONFIGURATION ──
        layer_weights = {}
        if ontology_file is not None:
            try:
                # Read and parse the uploaded ontology to extract layers
                ontology_content = json.loads(ontology_file.getvalue().decode('utf-8'))
                detected_layers = list(ontology_content.keys())
                
                if detected_layers:
                    st.markdown("#### 🎚️ Layer Priority Weights")
                    st.caption("Assign priority multipliers for each ontology layer (1.0 = neutral, higher = more important)")
                    
                    # Create weight inputs for each detected layer
                    cols_per_row = 3
                    for i in range(0, len(detected_layers), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(detected_layers):
                                layer_name = detected_layers[idx]
                                with col:
                                    weight = st.number_input(
                                        f"{layer_name}",
                                        min_value=0.0,
                                        max_value=10.0,
                                        value=1.0,
                                        step=0.1,
                                        key=f"weight_{layer_name}",
                                        help=f"Priority weight for {layer_name} layer"
                                    )
                                    layer_weights[layer_name] = weight
                    
                    st.markdown("---")
            except json.JSONDecodeError:
                st.warning("⚠️ Could not parse ontology file to detect layers. Using default weights.")
            except Exception as e:
                st.warning(f"⚠️ Error reading ontology: {e}")
        
        field_of_interest = st.text_input("Field of interest (e.g. materials science, biology, medicine)", value="materials science")
        
        # LLM selection with user-friendly display names
        llm_display_names = {
            "openAI": "OpenAI (gpt-4o)",
            "gemini": "Gemini (gemini-2.0-flash)",
            "llama": "Llama (llama-3.3)",
            "ollama": "Ollama (custom local LLM)",
            "claude": "Claude (claude-sonnet-4.0)"
        }
        agent = st.selectbox(
            "LLM", 
            options=["openAI", "gemini", "llama", "ollama", "claude"],
            format_func=lambda x: llm_display_names[x]
        )
        
        api_key_file = st.file_uploader("LLM API key (.txt)", type=["txt"]) if _needs_api_key(agent) else None
        keywords_file = st.file_uploader("Keyword Query file (.txt)", type=["txt"])
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
        if _needs_api_key(agent) and not api_key_file:
            missing_inputs.append(f"{agent} API key (.txt)")

        if not keywords_file:
            missing_inputs.append("Keywords file (.txt)")

        if _needs_kg_cred(kg_type) and not kg_credentials:
            missing_inputs.append("KG credentials (.json)")

        if missing_inputs:
            st.error("❌ Please upload or select the following before running:\n\n" + "\n".join(f"• {item}" for item in missing_inputs))
            st.stop()

        # Save uploads
        api_keys_path = _save_upload(api_key_file, ".txt") if api_key_file else None
        query_path   = _save_upload(query_file, ".txt")   if query_file   else None
        pdf_path     = _save_upload(pdf_zip,  ".zip")     if pdf_zip      else None
        keywords_path= _save_upload(keywords_file, ".txt")
        # ontology_path = Path("/Users/hrishilogani/Desktop/soke-test-data/Ontology.json")
        if ontology_file:
            ontology_path=_save_upload(ontology_file, ontology_file.name.split(".")[-1])
        creds_path = _save_upload(kg_credentials, ".json") if kg_credentials else None

        # 0. AI agent…
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

            # 3️⃣ Rank papers
            status.update(label="3/4 • Ranking papers…")
            t3_start = time.perf_counter(); w3_start = datetime.now()
            
            # Log layer weights being used
            if layer_weights:
                LOG.info(f"🎚️ Using custom layer priority weights: {layer_weights}")
            else:
                LOG.info("🎚️ Using default layer priority weights")
            
            # Pass custom layer weights if configured, otherwise None (uses defaults)
            output_csv_paths, output_paths_all = rank_papers(
                papers_path, 
                str(updated_ontology_path), 
                str(keywords_path), 
                ai_tool, 
                output_dir,
                layer_priority_weights=layer_weights if layer_weights else None
            )
            
            print(f"print ranked path : {papers}")
            print(f"ranked_path_files : {output_csv_paths}")
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
        # show_json_file(st.session_state["ontology_path"])
        _show_json_file_theme_safe(st.session_state["ontology_path"])
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
        
        # Use expander to prevent automatic re-rendering on every interaction
        with st.expander("View Full Knowledge Graph", expanded=False):
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

def _show_json_file_theme_safe(path: str):
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        st.json(obj, expanded=False)
    except Exception:
        # Fallback: still theme-safe because of our injected CSS on <pre>/<code>
        st.code(Path(path).read_text(encoding="utf-8"), language="json")

if __name__ == "__main__":
    main()