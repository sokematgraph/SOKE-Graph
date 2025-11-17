# 🧠 SOKE Graph: A Semantic-linked Ontological Framework for Domain-Specific Knowledge Discovery in Scientific Literature

SOKE Graph is a powerful, end-to-end pipeline designed to extract structured knowledge from scientific PDFs using ontology-driven classification and AI-assisted language models. It enables automated discovery and categorisation of domain-specific information—such as catalyst types, reaction conditions, and performance metrics—by parsing research papers, classifying concepts across multiple layers (e.g., Process, Environment, Reaction), and storing the results in a knowledge graph.

This tool can be tailored for accelerating literature analysis in any domain of research; in our case, we have focused on material science fields like green hydrogen production and water electrolysis.

## 📑 Table of Contents

- [🧠 SOKE Graph: A Semantic-linked Ontological Framework](#-soke-graph-a-semantic-linked-ontological-framework-for-domain-specific-knowledge-discovery-in-scientific-literature)
- [🚀 Features](#-features)
- [🚀 How to Run This Python Project on Windows, macOS, and Linux](#-how-to-run-this-python-project-on-windows-macos-and-linux)
  - [Step 1: Open the Command Line / Terminal](#step-1-open-the-command-line--terminal)
  - [Step 2: Clone the Project (Download the Code)](#step-2-clone-the-project-download-the-code)
  - [Step 3: Create a Virtual Environment](#step-3-create-a-virtual-environment-conda-recommended)
  - [Step 4: Activate the Virtual Environment](#step-4-activate-the-environment)
  - [Step 5: Install Project Dependencies](#step-5-install-project-dependencies)
  - [Step 6: Recommended Editor – VS Code](#step-6-recommended-editor--visual-studio-code-vs-code)
  - [Step 7: Run the Project](#step-7-run-the-project)
    - [1️⃣ Run with Streamlit App](#1️⃣-run-with-streamlit-app--streamlit-apppy)
    - [2️⃣ Run from Jupyter Notebook – full_pipeline.ipynb](#2️⃣-run-from-jupyter-notebook--full_pipelineipynb)
    - [3️⃣ Run from Jupyter Notebook – Step-by-Step](#3️⃣-run-from-jupyter-notebook-interactive-step-by-step--full_pipeline_stepbystepipynb)
- [📂 Preparing Input Files for SOKEGraph](#-preparing-input-files-for-sokegraph)
  - [1) Ontology File](#1-ontology-file-ontologyjson)
  - [2) Query File](#2-query-file-paper_querytxt)
  - [3) Keyword File](#3-keyword-file-keyword_querytxt)
  - [4) API Key File](#4-🔑-api-key-file-apikeys_xxxtxt)
- [Step 8: Deactivate Virtual Environment](#step-8-deactivate-virtual-environment-optional)
- [Reusing the Tool](#reusing-the-tool)

## 🚀 Features

- 🔍 **Retrieve papers** from Semantic Scholar or your PDF collection
- 🤖 **Use AI (OpenAI, Gemini, ...)** to extract ontological concepts and metadata
- 📊 **Rank papers** based on query relevance and extracted metadata
- 🧱 **Build knowledge graphs** (Neo4j and NetworkX supported) from structured paper data

---
# 🚀 How to Run This Python Project on Windows, macOS, and Linux

This guide will walk you through running this project on your computer, regardless of your operating system or prior Python knowledge.

---

## Step 1: Open the Command Line / Terminal

You'll be able to enter commands here.

- **Windows:** Press `Win + R`, type `cmd`, and press Enter to open Command Prompt.  
  Or, press `Win + X`, then select **Windows PowerShell** or **Windows Terminal** if installed.

- **macOS:** Press `Cmd + Space`, type `Terminal`, and press Enter.

- **Linux:** Look for the Terminal app in your applications menu, or press `Ctrl + Alt + T`.

---

## Step 2: Clone the Project (Download the Code)

In the command line window you opened, type:

```bash
git clone https://github.com/sokematgraph/SOKE-Graph.git
```

👉 If Git is not installed on your system, please see [INSTALLATION.md](INSTALLATION.md) for details.

After cloning, navigate into the project folder:
  ```bash
  cd SOKEGraph
  ```

## Step 3: Create a Virtual Environment (Conda Recommended)

We recommend using **Conda** (or Miniconda/Mamba) to manage dependencies for this project.  
Conda makes it easy to install and manage scientific packages across platforms.

```bash
conda create -n sokegraph python=3.9
```

---

## Step 4: Activate the Environment

```bash
conda activate sokegraph
```

You’ll need to activate this environment every time before running the project.

---
## Step 5: Install Project Dependencies

With the environment active, install required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

> ⚡ **Note for advanced users:**  
> You may also use Python’s built-in `venv` if you prefer, but Conda is the recommended and tested way for this project.

---


## Step 6: Recommended Editor – Visual Studio Code (VS Code)

We recommend using **Visual Studio Code (VS Code)** for working with this project, whether you want to edit code, run the Streamlit app, or work in Jupyter Notebooks.

### Installing VS Code
If you don’t already have VS Code installed, please see [INSTALLATION.md](INSTALLATION.md) for detailed instructions on how to download and install it.

### Opening the Project
You can open the `SOKEGraph` project folder in two ways:

- **Option 1: From VS Code directly**  
  - Open **VS Code**  
  - Go to **File > Open Folder...** and select the `SOKEGraph` folder  

- **Option 2: From the Terminal**  
  If VS Code is installed and added to your PATH, you can run:  
  ```bash
  cd SOKEGraph
  code .
  ```

After opening, use the integrated terminal (**View > Terminal**) to activate your virtual environment (see Step 4) and start running the project.

### Install VS Code Extensions
- Python (Microsoft)  
- Jupyter (Microsoft)  

These extensions make it easier to run and edit `.py` or `.ipynb` files directly inside VS Code.

💡 Tip: You can run Jupyter notebooks inside VS Code without opening a separate browser window.


## Step 7: Run the Project
You can choose the method that best fits your skills and setup. For most users, **Streamlit app is the easiest way** to get started.

---



### 1️⃣ Run with Streamlit App – `streamlit-app.py`

The Streamlit app provides a **simple graphical interface** to run the entire pipeline without writing code.  

#### How to start the app
From your project folder, run:

```bash
streamlit run streamlit-app.py
```

#### What you’ll see
The app will open in your browser. You can configure the pipeline with the following inputs:

- **Paper source**: Choose how to retrieve papers  
  - `Semantic Scholar`  
  - `Journal API`  
  - `PDF ZIP` (upload a ZIP file of papers)  

- **Number of papers**: The maximum number of papers to fetch (for Semantic Scholar or Journal API).  

- **Upload query file (`paper_query.txt`)**: A text file with one search query per line.  

- **Upload base ontology file (`Ontology.json`)**: Defines categories, subcategories, and keywords for concept detection.  

- **Field of interest**: Enter your research domain (e.g., *materials science, biology, medicine*).  

- **AI Agent**: Select which LLM to use for ontology enrichment and paper analysis (`openAI`, `gemini`, `llama`, `ollama`, or `claude`).  

- **Upload API key file (`apikeys_xxx.txt`)**: A text file containing the API keys required for accessing AI models and/or Journal APIs.  

- **Upload keywords file (`keyword_query.txt`)**: A list of keywords used for ranking and filtering papers.  
 

- **Knowledge Graph backend**: Choose the graph engine:  
  - `networkx` (in-memory, default)  
  - `neo4j` (requires credentials file)  

👉 **Note:** If you don’t know how to create these files (`Ontology.json`, `paper_query.txt`, `keyword_query.txt`, `apikeys_xxx.txt`, or `neo4j_credentials.json`), see the section [📂 Preparing Input Files for SOKEGraph](#-preparing-input-files-for-sokegraph) below. 

- **Compute Device**:  
  By default, the program runs with **GPU acceleration** if your system supports it (e.g., CUDA, MPS).  
  - To force the program to run on **CPU only**, check the option **“Force CPU (ignore GPU)”** in the sidebar.  

#### Running the pipeline
Once all inputs are set, click **🚀 Run Pipeline**.  

The app will:  
1. Fetch papers from your chosen source using queries.  
2. Enrich the ontology with AI.  
3. Rank the papers using keywords.  
4. Build and display the knowledge graph.  
5. Export results (ranked papers, ontology, graph data) into the `external/output/` folder.  


### 2️⃣ Run from Jupyter Notebook – `full_pipeline.ipynb`  
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


### 3️⃣ Run from Jupyter Notebook (Interactive Step-by-Step) — `full_pipeline_stepBYstep.ipynb`

This notebook uses **ipywidgets** to provide an **interactive form-like interface** for running the pipeline.  
It’s helpful if you want a guided, cell-by-cell execution **without writing code manually**.

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
  
---

# 📂 Preparing Input Files for SOKEGraph

SOKEGraph uses four input files. Place them in your project (e.g., `./inputs/`) and point the app/notebook to their paths.

## 1) 🧭 Ontology File (`Ontology.json`)
Defines **categories → subcategories → keywords/synonyms** that guide concept detection and search.

**Format**
```json
{
  "Category": {
    "Subcategory": ["keyword1", "keyword2", "keyword3"]
  }
}
```

**Example**
```json
{
  "Environment": {
    "Acidic": ["pH < 7", "acidic"],
    "Alkaline": ["pH > 7", "alkaline", "basic"]
  },
  "Process": {
    "Water Electrolysis": ["electrolysis of water", "splitting H2O"],
    "Fuel Cells": ["fuel cell", "PEM", "proton exchange membrane"]
  }
}
```

**Tips**
- Include common variants (symbols, abbreviations, spacing: `pH<7` vs `pH < 7`).
- Validate JSON (e.g., jsonlint). Save as `Ontology.json`.

## 2) 📄 Query File (`paper_query.txt`)
Each **line** is one search query sent to Semantic Scholar / other engines.

**Example**
```txt
Acidic earth abundant catalysts for water splitting
Nickel-based electrocatalysts for OER
Graph neural networks for chemical reaction prediction
```

## 3) 🔑 Keyword File (`keyword_query.txt`)
The `keyword_query.txt` file contains keywords or short phrases (e.g., *acidic HER water splitting*) that the system uses to **rank papers** during search.

**Example**
```txt
acidic HER water splitting
```

## 4) 🔑 API Key File (`apikeys_xxx.txt`)

The application requires **API keys** to access AI agents (OpenAI, Gemini, Claude, LLaMA, etc.) and external Journal APIs.  

### File Structure
- For each AI agent, you should create a **separate text file** (e.g., `openai_keys.txt`, `gemini_keys.txt`, `claude_keys.txt`, `llama_keys.txt`).  
- Each file can contain **multiple API keys**, one per line.  
- The application will automatically **iterate over these keys** if one is rate-limited or exhausted.  

**Example – `openai_keys.txt`**
```txt
sk-openai-xxxxxxxxxxxxxxxxxxxxxxxx
sk-openai-yyyyyyyyyyyyyyyyyyyyyyyy
```

**Example – `gemini_keys.txt`**
```txt
ya29.gemini-xxxxxxxxxxxxxxxx
ya29.gemini-yyyyyyyyyyyyyyyy
```

**Example – `claude_keys.txt`**
```txt
claude-xxxxxxxxxxxxxxxx
claude-yyyyyyyyyyyyyyyy
```

**Example – `llama_keys.txt`**
```txt
llama-xxxxxxxxxxxxxxxx
llama-yyyyyyyyyyyyyyyy
```

**Example – `journal_api_keys.txt`**
```txt
journal-abc123456789
journal-def987654321
```

---

### How to Create/Get API Keys

- **OpenAI**:  
  1. Sign up at [https://platform.openai.com](https://platform.openai.com).  
  2. Go to **View API keys**.  
  3. Create a new secret key and copy it into `openai_keys.txt`.  

- **Google Gemini (Vertex AI / Google AI Studio)**:  
  1. Go to [Google AI Studio](https://aistudio.google.com/) or [Google Cloud Console](https://console.cloud.google.com).  
  2. Enable Gemini API.  
  3. Generate an API key and add it to `gemini_keys.txt`.  

- **Anthropic Claude**:  
  1. Sign up at [https://console.anthropic.com](https://console.anthropic.com).  
  2. Generate an API key.  
  3. Save it into `claude_keys.txt`.  

- **Meta LLaMA (Together)**:  
  1. Go to Together: [Together AI](https://www.together.ai)
  2. Create an API key in the console. 
  3. Save it into `llama_keys.txt`.  

- **Journal API (e.g., Web of Science, Scopus, or other provider)**:  
  1. Log in to the provider’s portal.  
  2. Request an API token.  
  3. Save it into `journal_api_keys.txt`.  

- **Ollama**:  
  Ollama runs **offline locally** on your machine and **does not require an API key**. You just need to have Ollama installed and running.  

---

👉 Keep all API key files **private** and never commit them to GitHub.  
When running the app, simply **upload the relevant file(s)** in the Streamlit interface.  



---
## 5) 🗝️ Neo4j Credentials File (`neo4j_credentials.json`)

Provide your Neo4j connection details in a small JSON file.

**Example — neo4j_credentials.json**
```json
{
  "uri": "bolt://localhost:7687",
  "username": "neo4j",
  "password": "YOUR_PASSWORD",
}
```


**Recommended Layout**
```
inputs/
  Ontology.json
  paper_query.txt
  keyword_query.txt
  apikeys_xxx.txt
  neo4j_credentials.json

```

Point the Streamlit app / notebooks to these files when prompted.
---

## Step 8: Deactivate Virtual Environment (Optional)

When you are done working, you can leave the environment by running:

```bash
conda deactivate
```

👉 Whenever you want to use the tool again, just activate the environment:

```bash
conda activate sokegraph
```

Then run the project as shown in Step 7.
