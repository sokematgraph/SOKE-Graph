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

```bash
git clone https://github.com/your_username/sokegraph.git
cd sokegraph
```

### 2. Set up the Conda environment

```bash
conda create -n sokegraph python=3.9
conda activate sokegraph
pip install -r requirements.txt
```

---

## 🛠️ Configuration

You need to define your parameters, typically passed to the `full_pipeline_main(params)` function. Example parameters include:

```python
params = {
    "AI": "openAI",  # or "gemini"
    "API_keys": "openai_keys.json",
    "number_papers": 10,
    "paper_query_file": "topics.txt",
    "pdfs_file": None,  # if using PDF input instead
    "ontology_file": "base_ontology.json",
    "output_dir": "output/",
    "keyword_query_file": "keywords.txt",
    "credentials_for_knowledge_graph": "neo4j_credentials.json",
    "model_knowledge_graph": "neo4j"
}
```

---

## 📈 Usage

```python
from sokegraph.full_pipeline import full_pipeline_main

full_pipeline_main(params)
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

- Place your **OpenAI** or **Gemini** API keys in a JSON file: `keys/openai_keys.json`
- Neo4j credentials should be stored in: `config/neo4j_credentials.json`

Example:
```json
{
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_user": "neo4j",
  "neo4j_pass": "your_password"
}
```

---

## 🧪 Testing

Run the pipeline with test data:
```bash
sokegraph -n 200 -pq external/input/paper_query.txt -ky external/input/keyword_query.txt -ont external/input/Ontology.json -API external/input/APIs.txt -ckg external/input/neo4j.json -o external/output
```

---

## 🙋‍♀️ Contributions

Feel free to open issues or pull requests. Let’s improve this together!

---

## 📄 License

MIT License © 2025 [NRC]

---


