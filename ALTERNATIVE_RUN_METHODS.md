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