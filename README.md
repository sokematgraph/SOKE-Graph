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

💡 Make sure the paths are correct and required services (Neo4j, Ollama, etc.) are running.


### 2️⃣ Run from Jupyter Notebook (Interactive Step-by-Step) – `full_pipeline_stepBYstep.ipynb`

This notebook provides an **interactive UI-based form**, perfect for beginners who prefer not to write code manually.

#### 🧩 What it does:
- You choose how to input your papers:
  - Upload a ZIP file of PDFs **or**
  - Search papers from **Semantic Scholar** using a text file of queries.
  - Search papers from **Journal API** using a text file of queries and api key.
  
- You'll use dropdowns and file pickers to provide required files.
- You run the pipeline step by step to see exactly what happens at each stage.

#### 📋 Steps involved:
1. **Paper Retrieval**
   - If using Semantic Scholar, it downloads papers based on your query file.
   - If using a ZIP file, it extracts and analyzes the PDFs.

2. **Ontology Enrichment**
   - The AI agent (OpenAI, Gemini, LLaMA, or Ollama) enriches the ontology file with keywords and paper metadata.

3. **Paper Ranking**
   - Papers are scored based on keyword relevance, synonyms, and opposites to help identify the most relevant literature.

4. **Knowledge Graph Construction**
   - The enriched data is converted into a structured **Neo4j graph**, including categories, entities, and links between papers.

5. **Output**
   - All results (ontology, ranked papers, logs) are saved to your selected output directory.

> ✅ **No need to modify code manually**—just fill the form and click Submit.

> 💡 Make sure Neo4j or Ollama are running in the background if you're using them.

---


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
  -n 200 \                                    # --number_papers : how many papers to fetch
  -pdfs "" \                                  # --pdfs_file : ZIP of PDFs (leave empty if using -n)
  -pq external/input/paper_query.txt \        # --paper_query_file : search-query file
  -ky external/input/keyword_query.txt \      # --keyword_query_file : keywords for ranking
  -ont external/input/Ontology.json \         # --ontology_file : base ontology JSON
  -API external/input/API_keys.txt \          # --API_keys : file with your API tokens
  -ckg external/input/neo4j.json \            # --credentials_for_knowledge_graph : Neo4j creds
  -mkg neo4j \                                # --model_knowledge_graph : graph backend
  -AI openAI \                                # --AI : choose AI engine (openAI, gemini, llama…)
  -o external/output                          # --output_dir : where all results will be saved

```

---

#### 🧾 CLI Argument Explanation

| Flag    | Long Option                         | Example Value       | Purpose                                                                                    |
| ------- | ----------------------------------- | ------------------- | ------------------------------------------------------------------------------------------ |
| `-n`    | `--number_papers`                   | `200`               | Number of papers to fetch from Semantic Scholar. Use `0` or leave blank if supplying PDFs. |
| `-pdfs` | `--pdfs_file`                       | `papers.zip`        | ZIP file containing PDF papers for offline analysis. Leave empty when using `-n`.          |
| `-pq`   | `--paper_query_file`                | `paper_query.txt`   | Text file with search queries (one per line).                                              |
| `-ky`   | `--keyword_query_file`              | `keyword_query.txt` | Text file listing keywords used for ranking.                                               |
| `-ont`  | `--ontology_file`                   | `Ontology.json`     | JSON file containing the base ontology.                                                    |
| `-API`  | `--API_keys`                        | `API_keys.txt`      | File that stores your API keys (OpenAI, Gemini, etc.).                                     |
| `-ckg`  | `--credentials_for_knowledge_graph` | `neo4j.json`        | JSON with Neo4j URI + username + password.                                                 |
| `-mkg`  | `--model_knowledge_graph`           | `neo4j`             | Knowledge-graph backend (currently only `neo4j`).                                          |
| `-AI`   | `--AI`                              | `openAI`            | AI agent to use: `openAI`, `gemini`, `llama`, etc.                                         |
| `-o`    | `--output_dir`                      | `external/output`   | Folder where all pipeline outputs will be written.                                         |


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


