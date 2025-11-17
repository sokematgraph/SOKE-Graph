# ⚙️ Alternative Ways to Run SOKEGraph (Without Streamlit)

This guide explains how to run the SOKEGraph pipeline **without using the Streamlit app**. These methods are intended for users who want to:

- Work directly with the source code
- Use Jupyter Notebooks
- Modify pipeline parameters manually
- Integrate SOKEGraph into a larger workflow

Before using any of the methods below, **you must first complete Steps 1–5** from the main README:

1. Open Terminal
2. Clone the project
3. Create a virtual environment
4. Activate the environment
5. Install dependencies

See the main [README](README.md) for details.

---

## Step 6: Recommended Editor – Visual Studio Code (VS Code)

VS Code is highly recommended if you want to explore the codebase, run Jupyter notebooks, or make modifications to the pipeline.

### 🔧 Install VS Code
If you do not have VS Code installed, please follow the instructions in [INSTALLATION.md](INSTALLATION.md).

### 📂 Open the Project in VS Code

You can open the project folder using either method:

#### **Option 1: From VS Code**
1. Open **VS Code**
2. Go to **File > Open Folder...**
3. Select the `SOKE-Graph` folder

#### **Option 2: From Terminal**
If VS Code is added to your PATH:
```bash
code .
```

Once opened, use **View > Terminal** to activate your virtual environment:
```bash
conda activate sokegraph
```

---

## 🔌 VS Code Extensions to Install
- **Python** (Microsoft)
- **Jupyter** (Microsoft)

These extensions provide notebook support, IntelliSense, variable explorers, and inline execution.

💡 **Tip:** Jupyter notebooks run directly inside VS Code — no browser required.

---

# 1️⃣ Run from Jupyter Notebook — `full_pipeline.ipynb`

This notebook is ideal for users who prefer **script-like control** without needing a UI.

### What this notebook offers
- Configure the entire pipeline in one place
- Run everything with a **single function call**
- Perfect for experiments, research workflows, and debugging

### 🧪 Example Usage
```python
from types import SimpleNamespace
from sokegraph.full_pipeline import full_pipeline_main

params = SimpleNamespace(
    paper_source="Semantic Scholar",
    number_papers=10,
    paper_query_file="topics.txt",
    pdfs_file=None,
    api_key_file="api_journal_api.txt",
    ontology_file="base_ontology.json",
    field_of_interest="material sciences",
    AI="openAI",
    API_keys="openai_keys.json",
    keyword_query_file="keywords.txt",
    model_knowledge_graph="networkx",
    credentials_for_knowledge_graph="neo4j_credentials.json",
    output_dir="output/"
)

full_pipeline_main(params)
```

### ⚠️ Important Rules
Depending on your workflow, use:
- `number_papers` + `paper_query_file`
- `number_papers` + `paper_query_file` + `api_key_journal_api`
- **OR** `pdfs_file` (for PDF uploads)

Make sure:
- All file paths are valid.
- Neo4j/Journals/Ollama are running if you depend on them.

---

# 2️⃣ Run from Jupyter Notebook (Interactive Step-by-Step) — `full_pipeline_stepBYstep.ipynb`

This notebook is ideal for users who want a **guided, form-like interface** without writing code.

### 🧩 What this notebook provides
- File pickers for uploading ontology, queries, API keys, and PDFs
- Options for selecting paper sources and AI agents
- Dropdown menus for graph backend selection
- Buttons for running each pipeline stage interactively

### 📋 Pipeline Steps (Interactive)
1. **📄 Retrieve Papers**
2. **🧠 Enrich Ontology using AI**
3. **📊 Rank Papers using Keywords + Semantic Expansion**
4. **🕸 Build Knowledge Graph (Neo4j or NetworkX)**
5. **💾 Save Results to Output Folder**

---

> ✅ **No need to modify code manually** — select options, upload files, and click **Run**.

> 💡 Ensure external services (Neo4j, Ollama, Journal APIs) are available before starting.