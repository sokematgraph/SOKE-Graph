"""streamlit_app.py – Streamlit front‑end wrapper for SOKEGraph

▶️ **Change**: Output directory is now selected using a file system directory picker.
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace

import streamlit as st


# --- Import your existing backend -------------------------------------------------
from sokegraph.semantic_scholar_source import SemanticScholarPaperSource
from sokegraph.pdf_paper_source import PDFPaperSource
from sokegraph.paper_ranker import PaperRanker  # type: ignore
from sokegraph.neo4j_knowledge_graph import Neo4jKnowledgeGraph  # type: ignore
from sokegraph.ontology_updater import OntologyUpdater  # type: ignore
from sokegraph.full_pipeline import full_pipeline_main  # make sure this exists in your package

# ----------------------------------------------------------------------------------

def _needs_api_key(agent: str) -> bool:
    return agent.lower() in {"openai", "gemini", "llama"}



def _save_upload(uploaded_file, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    return Path(tmp.name)



# -----------------------------------------------------------------------------
# Streamlit UI -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="SOKEGraph", layout="wide")
    st.title("📚 SOKEGraph – Paper Retrieval & Knowledge Graph Builder")

    left, right = st.columns([1.2, 2])

    # ---------------------------------------------------------------------
    # LEFT COLUMN – PAPER SOURCE
    # ---------------------------------------------------------------------
    with left:
        st.subheader("📂 Select Paper Source")
        paper_mode = st.radio("Choose source:", ["Semantic Scholar", "PDF ZIP"], index=0)

    # ---------------------------------------------------------------------
    # RIGHT COLUMN – CONFIGURATION
    # ---------------------------------------------------------------------
    with right:
        st.subheader("⚙️ Configuration")

        # ------------------------------ Source‑specific inputs
        if paper_mode == "Semantic Scholar":
            num_papers = st.number_input("Number of papers", 1, 200, 10)
            query_file = st.file_uploader("Upload query file (.txt)", type=["txt"])
            pdf_zip = None
        else:
            pdf_zip = st.file_uploader("Upload PDF ZIP", type=["zip"])
            query_file = None
            num_papers = None

        # ------------------------------ Common files
        keywords_file = st.file_uploader("Upload keywords file (.txt)", type=["txt"])
        ontology_file = st.file_uploader("Upload ontology file (.json)", type=["json"])

        # ------------------------------ AI Agent + API key
        agent = st.selectbox("AI Agent:", ["openAI", "gemini", "llama", "ollama"])
        api_key_file = (
            st.file_uploader("Upload API Key file (.txt / .json)", type=["txt", "json"])
            if _needs_api_key(agent)
            else None
        )

        # ------------------------------ KG credentials (.json)
        kg_type = st.selectbox("Knowledge Graph:", ["neo4j"])
        kg_credentials = st.file_uploader("Upload KG credentials (.json)", type=["json"])

        # Output directory picker ------------------------------------------------
        output_dir = Path("external/output")

        output_dir.mkdir(parents=True, exist_ok=True)



        run_btn = st.button("🚀 Run Pipeline")

    # ---------------------------------------------------------------------
    # EXECUTION
    # ---------------------------------------------------------------------
    if run_btn:
        errs = []
        if _needs_api_key(agent) and api_key_file is None:
            errs.append("❌ API key file is required for the selected agent.")
        if keywords_file is None:
            errs.append("❌ Please upload a keywords file (.txt).")
        if ontology_file is None:
            errs.append("❌ Please upload an ontology file (.json).")
        if kg_credentials is None:
            errs.append("❌ Please upload KG credentials (.json).")
        if paper_mode == "Semantic Scholar" and query_file is None:
            errs.append("❌ Please upload a search‑query file (.txt).")
        if paper_mode == "PDF ZIP" and pdf_zip is None:
            errs.append("❌ Please upload a ZIP file containing PDFs.")
        

        if errs:
            st.error("\n".join(errs))
            st.stop()

        # Save uploads
        api_key_path = _save_upload(api_key_file, ".txt") if api_key_file else None
        query_path = _save_upload(query_file, ".txt") if query_file else None
        pdf_path = _save_upload(pdf_zip, ".zip") if pdf_zip else None
        keywords_path = _save_upload(keywords_file, ".txt")
        ontology_path = _save_upload(ontology_file, ".json")
        creds_path = _save_upload(kg_credentials, ".json")

        params = SimpleNamespace(
            AI = str(agent),
            API_keys= str(api_key_path) if api_key_path else None,
            number_papers= int(num_papers) if num_papers else None,
            paper_query_file= str(query_path) if query_path else None,
            pdfs_file= str(pdf_path) if pdf_path else None,
            ontology_file= str(ontology_path),
            output_dir= str(output_dir),
            keyword_query_file= str(keywords_path),
            credentials_for_knowledge_graph= str(creds_path),
            model_knowledge_graph= kg_type,
        )

        with st.spinner("🔁 Running Full Pipeline …"):
            result = full_pipeline_main(params)

        st.success("✅ Pipeline completed.")
        st.write("Returned:", result)


if __name__ == "__main__":
    main()
