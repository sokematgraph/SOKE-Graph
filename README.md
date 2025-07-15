# 🧠 SOKE Graph: A Semantic-linked Ontological Framework for Domain-Specific Knowledge Discovery in Scientific Literature

SOKE Graph is a powerful, end-to-end pipeline designed to extract structured knowledge from scientific PDFs using ontology-driven classification and AI-assisted language models. It enables automated discovery and categorization of domain-specific information—such as catalyst types, reaction conditions, and performance metrics—by parsing research papers, classifying concepts across multiple layers (e.g., Process, Environment, Reaction), and storing the results in a knowledge graph.

This tool can be tailored for accelerating literature analysis in any domain of research; in our case, we have focused on material science fields like green hydrogen production and water electrolysis.


## 🚀 Features

- 🔍 **Retrieve papers** from Semantic Scholar or your own PDF collection
- 🤖 **Use AI (OpenAI or Gemini)** to extract ontological concepts and metadata
- 📊 **Rank papers** based on query relevance and extracted metadata
- 🧱 **Build knowledge graphs** (Neo4j supported) from structured paper data

---

## 📦 Installation

### 1. Clone the repository

### ✅ 1. Get the Code

Open a terminal (or use Google Colab) and run:

```bash
git clone https://github.com/sanakashgouli/SOKEGraph.git
cd sokegraph
```

> 🔒 If the repo is private and you're using Google Colab, use a [GitHub token](https://github.com/settings/tokens) to authenticate.

### ✅ 2. Set up the Conda environment

```bash
conda create -n sokegraph python=3.9
conda activate sokegraph
pip install -r requirements.txt
```
---

## 🚀 How to Run SOKEGraph

You can run the full pipeline in **four different ways** depending on your preference and setup:

---

### 1️⃣ Run from Jupyter Notebook – `full_pipeline.ipynb`  
This notebook is designed for users who are comfortable modifying code directly.

- 🔧 You should define all parameters manually in a Python dictionary called `params`.
- ✅ Once configured, you run the pipeline **with a single function call**.
- 📂 Best for quick experiments or automation in notebook environments.

#### Example Usage:

```python
from types import SimpleNamespace
from sokegraph.full_pipeline import full_pipeline_main

params = SimpleNamespace(
    paper_source="Semantic Scholar",  # Options: "Semantic Scholar", "PDF Zip", "Journal API"
    number_papers=10,                # Number of papers to fetch from Semantic Scholar
    paper_query_file="topics.txt",   # Text file with one search query per line
    pdfs_file=None,                  # Optional: ZIP file with PDFs (for PDF source)
    api_key_file="api_journal_api.txt",  # API key file for Journal API source
    ontology_file="base_ontology.json",  # Base ontology file (JSON or OWL)
    AI="openAI",                          # Options: "openAI", "gemini", "llama", "ollama", "claude"
    API_keys="openai_keys.json",         # API key file for AI tools
    keyword_query_file="keywords.txt",   # Text file listing keywords
    model_knowledge_graph="neo4j",       # Options: "neo4j", "networkx"
    credentials_for_knowledge_graph="neo4j_credentials.json",  # Graph DB credentials
    output_dir="output/"                 # Output directory
)

full_pipeline_main(params)
```

⚠️ Important: You should use either:

"number_papers" + "paper_query_file"

OR

"number_papers" + "paper_query_file" + "api_key_journal_api"

OR

"pdfs_file"

depending on whether you're searching for papers or uploading PDFs.

💡 Make sure that all file paths in your `params` are valid and that services like **Neo4j**, **Ollama**, or your Journal API access are available before starting the pipeline.


### 2️⃣ Run from Jupyter Notebook (Interactive Step-by-Step) – `full_pipeline_stepBYstep.ipynb`

This notebook offers an **interactive, beginner-friendly form-based UI**, ideal for users who want to run the pipeline without writing code.

#### 🧩 What it does:
- Allows you to select how you want to retrieve papers:
  - 📁 Upload a ZIP file of PDFs (PDF source)
  - 🔎 Search and fetch papers from **Semantic Scholar** using a query file
  - 🌐 Fetch papers via the **Journal API** using a query file and an API key

- Provides dropdowns and file pickers to easily select files like:
  - Ontology
  - Keyword queries
  - API keys
  - Output folder

- Runs each pipeline step independently, so you can see exactly what happens at every stage.

#### 📋 Steps Involved:

1. **📄 Paper Retrieval**
   - Based on your selected `paper_source`:
     - `Semantic Scholar`: Downloads papers using your `paper_query_file`
     - `PDF Zip`: Loads and processes PDFs from the uploaded ZIP file
     - `Journal API`: Retrieves paper metadata from the Web of Science API using query + API key

2. **🧠 Ontology Enrichment**
   - The chosen AI agent (`openAI`, `gemini`, `llama`, `ollama`, or `claude`) analyzes the papers and expands your base ontology
   - Adds new keywords, concepts, synonyms, and relationships

3. **📊 Paper Ranking**
   - Ranks the papers using:
     - Exact keyword matches
     - Synonyms and expanded terms
     - Opposite-term filtering to down-rank irrelevant papers

4. **🕸 Knowledge Graph Construction**
   - Converts enriched data into a structured graph using:
     - `Neo4j` (with login credentials)
     - Or `NetworkX` (in-memory option)
   - Graph includes:
     - Ontology categories
     - Paper-concept links
     - Metadata associations

5. **💾 Output**
   - Saves everything in your selected `output_dir`, including:
     - Enriched ontology file
     - Ranked papers (CSV/JSON)

---

> ✅ **No need to modify code manually** – just fill out the form and click **Run** for each step.

> 💡 Make sure required services like **Neo4j**, **Ollama**, or your **Journal API** credentials are ready before starting the pipeline.


### 3️⃣ Run from Command Line 

This option is great for developers or users who prefer terminal-based execution and want to run the entire pipeline using a single command with a config dictionary.

---

#### ✅ Step 1: Install the Project Locally

From the root directory of the cloned repo:

```bash
pip install -e .
```

This installs the `sokegraph` command-line tool in your environment.

---

#### 📦 Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

#### 🚀 Step 3: Run the Pipeline from Terminal

```bash
sokegraph \
  -n 200 \                                      # --number_papers: number of papers to fetch from Semantic Scholar
  -pdfs "" \                                    # --pdfs_file: ZIP file with PDFs (leave empty if using -n)
  -api_journal external/input/journal_api_key.txt \  # --api_key_journal_api: API key file for Journal API (if used)
  -pq external/input/paper_query.txt \          # --paper_query_file: text file with search queries (one per line)
  -ky external/input/keyword_query.txt \        # --keyword_query_file: keywords for ranking and filtering
  -ont external/input/ontology.json \           # --ontology_file: base ontology file (JSON or OWL)
  -API external/input/API_keys.txt \            # --API_keys: file with your AI provider's API credentials
  -ckg external/input/neo4j.json \              # --credentials_for_knowledge_graph: Neo4j credentials (JSON)
  -mkg neo4j \                                   # --model_knowledge_graph: choose 'neo4j' or 'networkx'
  -AI openAI \                                   # --AI: choose your AI engine (openAI, gemini, llama, etc.)
  -o external/output/                            # --output_dir: directory to store all outputs

```

---

#### 🧾 CLI Argument Explanation

| Flag         | Long Option                          | Example Value                  | Purpose                                                                                             |
|--------------|--------------------------------------|--------------------------------|-----------------------------------------------------------------------------------------------------|
| `-n`         | `--number_papers`                    | `200`                          | Number of papers to fetch from Semantic Scholar or Journal API. Leave blank if using PDFs.          |
| `-pdfs`      | `--pdfs_file`                        | `papers.zip`                   | ZIP file containing PDF papers. Leave blank when using `-n`.                                        |
| `-api_journal` | `--api_key_journal_api`            | `journal_api_key.txt`          | File containing Journal API key (used when paper source is Journal API).                           |
| `-pq`        | `--paper_query_file`                 | `paper_query.txt`              | File with one search query per line (used by Semantic Scholar or Journal API).                     |
| `-ky`        | `--keyword_query_file`               | `keyword_query.txt`            | Keywords used for paper ranking and ontology enrichment.                                            |
| `-ont`       | `--ontology_file`                    | `ontology.json`                | Base ontology file (in JSON or OWL format).                                                         |
| `-API`       | `--API_keys`                         | `API_keys.txt`                 | File containing your AI provider API credentials (OpenAI, Gemini, etc.).                           |
| `-ckg`       | `--credentials_for_knowledge_graph`  | `neo4j.json`                   | Required only for Neo4j: JSON file with URI, username, and password.                               |
| `-mkg`       | `--model_knowledge_graph`            | `neo4j` or `networkx`          | Backend for the knowledge graph: use `neo4j` for DB or `networkx` for local graph construction.     |
| `-AI`        | `--AI`                               | `openAI`                       | AI engine to use: `openAI`, `gemini`, `llama`, `ollama`, or `claude`.                              |
| `-o`         | `--output_dir`                       | `external/output/`             | Directory where all outputs (ontology, rankings, logs) will be saved.                              |
| `--verbose`  | `--verbose`                          | *(flag only)*                  | Enables debug logging and keeps intermediate files.                                                 |
| `-f`         | `--force`                            | *(flag only)*                  | Overwrites existing outputs and ignores previous runs.                                              |



---

💡 **Tip:** Ensure your file paths exist and any external services like Neo4j or Ollama are running before execution.

### 🦙 Using Ollama Locally
#### ✅ Step 1: Install Ollama
Install Ollama from the official site or using the command line:

Website (all OS): https://ollama.com/download

macOS (via Homebrew):
```bash
brew install ollama
```

Linux (via script):
```bash
curl -fsSL https://ollama.com/install.sh | sh
```


#### 🚀 Step 2: Pull and Run the Model
Once installed, pull the model you want and run the Ollama server locally:
```bash
ollama run llama3
```

### Pipeline Steps:
1. **Initialize AI agent** (OpenAI or Gemini)
2. **Fetch papers** (from Semantic Scholar or a zip of PDFs)
3. **Update ontology** with AI-based enrichment
4. **Rank papers** based on extracted metadata and keywords
5. **Build knowledge graph** (Neo4j)

---

## 📂 Output

The pipeline will generate:
- `updated_ontology.json`: AI-enriched ontology
- `ranked_papers.csv`: Paper ranking results
- Neo4j graph: If configured with valid credentials

---

## 🔐 API Keys & Credentials

- Place your **OpenAI** or **Gemini** API keys in a JSON file: `ai_keys.json`
- Neo4j credentials should be stored in: `neo4j_credentials.json`

Example:
```json
{
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_user": "neo4j",
  "neo4j_pass": "your_password"
}
```

---

## 🙋‍♀️ Contributions

Feel free to open issues or pull requests. Let’s improve this together!

---

## 📄 License

MIT License © 2025 [NRC]

---


