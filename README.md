# 🧠 SOKE Graph: A Semantic-linked Ontological Framework for Domain-Specific Knowledge Discovery in Scientific Literature

SOKE Graph is a powerful, end-to-end pipeline designed to extract structured knowledge from scientific PDFs using ontology-driven classification and AI-assisted language models. It enables automated discovery and categorisation of domain-specific information—such as catalyst types, reaction conditions, and performance metrics—by parsing research papers, classifying concepts across multiple layers (e.g., Process, Environment, Reaction), and storing the results in a knowledge graph.

This tool can be tailored for accelerating literature analysis in any domain of research; in our case, we have focused on material science fields like green hydrogen production and water electrolysis.


## 🚀 Features

- 🔍 **Retrieve papers** from Semantic Scholar or your PDF collection
- 🤖 **Use AI (OpenAI, Gemini, ...)** to extract ontological concepts and metadata
- 📊 **Rank papers** based on query relevance and extracted metadata
- 🧱 **Build knowledge graphs** (Neo4j supported) from structured paper data

---
# 🚀 How to Run This Python Project on Windows, macOS, and Linux

This guide will walk you through running this project on your computer, regardless of your operating system or prior Python knowledge.

---

## Step 1: Install Python

Python is the programming language this project uses.

- **Download Python 3.x here:** https://www.python.org/downloads/  
- Make sure you download **Python 3** (not Python 2).

### Windows Users
- When installing Python, **make sure to check the box "Add Python to PATH"** on the installer screen!  
- If you forget this step, your computer won’t recognize Python commands.

### macOS Users
- Python 3 is often pre-installed, but it may be an older version.  
- Recommended: install the latest Python 3 via [Homebrew](https://brew.sh/):
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  brew install python
  ```
### Linux Users

- Most Linux distributions come with Python installed.

- To install or upgrade, use your package manager.

- For Ubuntu/Debian:

  ```bash
  sudo apt update
  sudo apt install python3 python3-venv python3-pip
  ```
## Step 2: Open the Command Line / Terminal

You'll be able to enter commands here.

- **Windows:** Press `Win + R`, type `cmd`, and press Enter to open Command Prompt.  
  Or, press `Win + X`, then select **Windows PowerShell** or **Windows Terminal** if installed.

- **macOS:** Press `Cmd + Space`, type `Terminal`, and press Enter.

- **Linux:** Look for the Terminal app in your applications menu, or press `Ctrl + Alt + T`.

---

## Step 3: Clone the Project (Download the Code)

In the command line window you opened, type:

```bash
git clone https://github.com/sanakashgouli/SOKEGraph.git
```

If you get an error saying `git` is not found:

- **Windows:** Download and install [Git for Windows](https://git-scm.com/download/win).

- **macOS:** Install via Homebrew with:

  ```bash
  brew install git
  ```

- **Linux:** Install with:

  ```bash
  sudo apt install git
  ```
  (For Ubuntu/Debian, or use your distro’s package manager.)

  After cloning, navigate into the project folder:
  ```bash
  cd SOKEGraph
  ```

## Step 3.1: Open the Project in VS Code

We recommend working with **Visual Studio Code (VS Code)** for this project.  
You can open the project in two ways:

### Option 1: Open from VS Code
- Open **VS Code**  
- Go to **File > Open Folder...** and select the `SOKEGraph` folder  

### Option 2: Open from Terminal
If VS Code is installed and added to your PATH, you can run:
```bash
cd SOKEGraph
code .
```

After opening, use the integrated terminal (View > Terminal) to activate your virtual environment (see Step 5) and start running the project.

---

## Step 4: Create a Virtual Environment (Isolate Dependencies)

This keeps the project’s Python packages separate from your system.

### Windows Users

In Command Prompt or PowerShell, run:

```powershell
python -m venv venv
```
If python is not recognized, try:

```powershell
py -3 -m venv venv
```
### macOS / Linux Users
Run:

```bash
python3 -m venv venv
```

---

## Step 5: Activate the Virtual Environment

You'll need to activate this environment every time before you run the project.

### Windows Users

- In Command Prompt, run:
  
```cmd
venv\Scripts\activate
```
- In PowerShell:
  
```powershell
venv\Scripts\Activate.ps1
```
> If you see an error about script execution policies in PowerShell:
Run this command once to allow scripts:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate again.
### macOS / Linux Users
Run:
```bash
source venv/bin/activate
```

---

## Step 6: Install Project Dependencies
With the virtual environment active (you should see (venv) at the start of your prompt), install packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 7: Recommended Editor – Visual Studio Code (VS Code)

We recommend using **VS Code** for working with this project, whether you want to edit code, run the Streamlit app, or work in Jupyter Notebooks.

- **Download**: [https://code.visualstudio.com](https://code.visualstudio.com)
- **Install Extensions**:
  - Python (Microsoft)
  - Jupyter (Microsoft)
- **Open the project folder** in VS Code and:
  - Use **Jupyter Notebook support** for `.ipynb` files
  - Use the integrated terminal to activate your virtual environment
- **Tip:** You can run Jupyter notebooks inside VS Code without opening a separate browser window.

---

## Step 8: Run the Project
You can choose the method that best fits your skills and setup. For most users, **Streamlit app is the easiest way** to get started.

---

### 1️⃣ Run with Streamlit App – `streamlit-app.py`

A simple, user-friendly graphical interface that lets you run the whole pipeline **without writing any code**.

#### How to run:

```bash
streamlit run streamlit-app.py
```
- Choose your **paper source**:  
  `Semantic Scholar`, `PDF ZIP`, or `Journal API`.

- Upload or select required files:
  - Paper queries
  - Ontology file
  - Keywords list
  - API keys (AI + Journal API)
  - Neo4j credentials (optional)

- Select an AI agent:
  - `openAI`, `gemini`, `llama`, `ollama`, or `claude`

- Choose your knowledge graph backend:
  - `neo4j` or `networkx`

- Run the full pipeline:
  - **Fetch → Enrich → Rank → Build Graph**

- View logs, ranked papers, and results directly in the UI.

---

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

## Step 9: Deactivate Virtual Environment (Optional)
When done working, you can leave the environment by running:
```bash
deactivate
```
---
## Reusing the Tool
If you have cloned the repository and want to use it again:

### Windows Users

In Command Prompt or PowerShell, run:

```bash
cd SOKEGraph
venv\Scripts\activate
streamlit run streamlit-app.py
```
