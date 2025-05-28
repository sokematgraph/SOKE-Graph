from semanticscholar import SemanticScholar
from time import sleep
import os
import json
import re
from openai import OpenAI
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Set
import random
import pandas as pd
import json
from py2neo import Graph, Node, Relationship
from collections import defaultdict
from itertools import combinations

import re


counter_index_API = 0

def full_pipeline_main(args):
    
    updated_ontology_file_path = ranking_papers(args.API_keys,
    args.ontology_file,
    int(args.number_papers),
    args.paper_query_file,
    args.keyword_query_file,
    args.output_dir)

    with open(args.credentials_for_knowledge_graph, "r") as file:
        credentials = json.load(file)

    build_knowledge_graph_from_ontology_json(updated_ontology_file_path,
    credentials["neo4j_uri"],
    credentials["neo4j_user"],
    credentials["neo4j_pass"])


def safe_title(title: str, max_len: int = 100) -> str:
    """
    Sanitize a string to create a safe title by removing special characters 
    and trimming it to a maximum length.

    Args:
        title (str): The original title string.
        max_len (int, optional): The maximum allowed length of the sanitized title. Default is 100.

    Returns:
        str: A cleaned and trimmed version of the title, containing only 
             alphanumeric characters, underscores, hyphens, and spaces.
    """
    # Remove characters not in the allowed set: letters, numbers, underscores, hyphens, and spaces
    cleaned_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title)
    
    # Truncate to the specified max_len and strip leading/trailing spaces
    return cleaned_title[:max_len].strip()


def get_unique_papers_from_sch(max_total: int, paper_query_file_path) -> List[Dict]:
    """
    Search Semantic Scholar for research papers matching a list of queries,
    and return a deduplicated list of paper metadata.

    Args:
        max_total (int): The maximum number of papers to retrieve in total.
        search_queries_list (List[str]): A list of search query strings.

    Returns:
        List[Dict]: A list of unique papers, where each paper is represented as a dictionary
                    containing keys: paper_id, title, abstract, authors, year, venue, url, and doi.
    """
    search_queries_list = load_queries_list(paper_query_file_path)


    # Initialize the Semantic Scholar API
    sch = SemanticScholar()
    
    # List to store all retrieved paper metadata
    semantic_scholar_papers = []

    # Loop through each query in the list
    for query in search_queries_list:
        # Stop fetching if we've reached the maximum allowed number of papers
        if len(semantic_scholar_papers) >= max_total:
            break

        print(f"🔍 Searching Semantic Scholar for: {query}")
        
        try:
            # Fetch up to 100 papers for the current query
            results = sch.search_paper(query, limit=100)

            # Process each paper
            for paper in results:
                if len(semantic_scholar_papers) >= max_total:
                    break

                semantic_scholar_papers.append({
                    "paper_id": paper.paperId,
                    "title": paper.title,
                    "abstract": paper.abstract or "",
                    "authors": ", ".join([a["name"] for a in paper.authors]) if paper.authors else "",
                    "year": paper.year,
                    "venue": paper.venue,
                    "url": paper.url,
                    "doi": paper.externalIds.get("DOI") if paper.externalIds else ""
                })

            # Be polite to the API
            sleep(1)

        except Exception as e:
            print(f"❌ Error searching '{query}': {e}")
            continue

    # Remove duplicate papers based on their paper ID
    unique_papers = {p['paper_id']: p for p in semantic_scholar_papers}
    semantic_scholar_papers = list(unique_papers.values())

    print(f"✅ Retrieved {len(semantic_scholar_papers)} unique papers.")


    text_data = {}
    title_map = {}
    abstract_map = {}

    for paper in semantic_scholar_papers:
        safe_id = safe_title(paper['title'] or paper['paper_id'])
        text_data[safe_id] = paper['abstract'] or ""
        title_map[safe_id] = paper['title'] or ""
        abstract_map[safe_id] = paper['abstract'] or ""

    print(f"✅ Organized {len(text_data)} papers with title-based IDs.")

    export_semantic_scholar_metadata_to_excel(semantic_scholar_papers)
    return semantic_scholar_papers


def load_queries_list(paper_query_file_path):
    with open(paper_query_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_api_keys(API_file_path: str) -> List[str]:
    """
    Load API keys from a text file, one key per line.

    Args:
        API_file_path (str): Path to the file containing API keys.

    Returns:
        List[str]: A list of non-empty, stripped API keys.
    """
    # Open the file and read all non-empty lines, stripping whitespace
    with open(API_file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

####check is it true
def get_next_api_key(api_keys):
    global counter_index_API
    counter_index_API = counter_index_API + 1
    key = api_keys[(counter_index_API + 1) % len(api_keys)]
    return key


#def get_next_api_key(api_keys: List[str]) -> Optional[str]:
    """
    Select and return a random API key from the provided list.

    Args:
        api_keys (List[str]): A list of available API keys.

    Returns:
        Optional[str]: A randomly selected API key, or None if the list is empty.
    """
#    if not api_keys:
#       print("No API keys available.")
#       return None

    # Return a randomly selected API key
#    return random.choice(api_keys)


def call_openai_model(layer_name: str, abstract_text: str, ontology_layer: Dict, api_keys: List[str]) -> List[Dict]:
    """
    Calls the OpenAI GPT-4o model to extract structured information from an abstract
    based on a specific ontology layer in materials science.

    Args:
        layer_name (str): The name of the ontology layer (e.g., "Catalyst", "Electrolyte").
        abstract_text (str): The scientific abstract or paragraph to analyze.
        ontology_layer (Dict): Dictionary of categories and example keywords for the specified layer.
        api_keys (List[str]): List of available OpenAI API keys.

    Returns:
        List[Dict]: A list of extracted keyword entries, each containing:
            - layer (str): The name of the ontology layer.
            - matched_category (str): The matched category from the ontology.
            - keyword (str): The extracted keyword or phrase.
            - meta_data (str): A surrounding text snippet for context.
    """
    # Initialize the OpenAI client with a random API key
    client = OpenAI(api_key=get_next_api_key(api_keys))

    # Construct the prompt for the language model
    prompt = f'''
        You are a structured information extraction AI specialized in materials science.

        You will extract keywords relevant to the layer: **{layer_name}**
        For this layer, the following categories and sample keywords are provided:
        {json.dumps(ontology_layer, indent=2)}

        From the text below, extract:
        - Any new or related keywords
        - The category it fits under
        - Any nearby numeric values with units (e.g., overpotential, pH, current density)

        Text to analyze:
        {abstract_text}

        Return the output in this exact JSON list format:
        [
        {{
            "layer": "LayerName",
            "matched_category": "Category",
            "keyword": "ExtractedKeyword",
            "meta_data": "Text snippet"
        }},
        ...
        ]
    '''

    try:
        # Send the prompt to the OpenAI model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a structured information extraction AI specialized in materials science."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        # Get and preview model output
        model_output = response.choices[0].message.content.strip()
        print("📄 OpenAI returned:\n", model_output[:500])  # Preview first 500 characters

        # Try parsing the output as JSON
        try:
            parsed = json.loads(model_output)
            print(f"✅ JSON parsed successfully. Extracted {len(parsed)} items.")
            return parsed
        except json.JSONDecodeError:
            print("⚠️ JSON parsing failed. Attempting fallback regex...")

            # Fallback to regex-based parsing if JSON fails
            pattern = r'"layer":\s*"([^"]+)",\s*"matched_category":\s*"([^"]+)",\s*"keyword":\s*"([^"]+)",\s*"meta_data":\s*"([^"]+)"'
            matches = re.findall(pattern, model_output, re.DOTALL)
            extracted = []
            for m in matches:
                extracted.append({
                    "layer": m[0].strip(),
                    "matched_category": m[1].strip(),
                    "keyword": m[2].strip(),
                    "meta_data": m[3].strip()
                })

            print(f"✅ Extracted {len(extracted)} items via regex.")
            return extracted

    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return []



def enrich_ontology_with_papers(ontology_path: str, api_keys_path: str, max_papers: int, paper_query_file_path, output_dir) -> None:
    """
    Enrich the ontology by extracting relevant keywords from paper abstracts using the OpenAI API,
    and save the enriched ontology structure to a JSON file.

    Args:
        ontology_path (str): Path to the JSON file containing the ontology structure.
        api_keys_path (str): Path to a text or JSON file containing OpenAI API keys.
        max_papers (int): Maximum number of papers to retrieve and analyze.

    Returns:
        None. The enriched ontology is saved to 'updated_ontology.json'.
    """

    # Step 1: Load the ontology from the file path, or use a predefined fallback if not available
    ontology = load_or_initialize_ontology(ontology_path)

    # Step 2: Load available API keys from the provided file
    api_keys = load_api_keys(api_keys_path)
    print("Total API keys loaded:", len(api_keys))

    # Step 3: Retrieve research papers (titles and abstracts), up to `max_papers`
    text_data, title_map, abstract_map = load_paper_data(max_papers, paper_query_file_path)
    print(f"✅ Organized {len(text_data)} papers with title-based IDs.")

    # Step 4: Extract relevant keywords from each paper abstract for each ontology layer
    ontology_extractions = extract_keywords(ontology, text_data, api_keys)

    # Step 5: Parse metadata (e.g., extract numerical values, trends) from each keyword match
    parse_all_metadata(ontology_extractions)

    # Step 6: Save the updated ontology structure to a JSON file
    updated_ontology_file_path = f"{output_dir}/updated_ontology.json"
    with open(updated_ontology_file_path, "w") as f:
        json.dump(ontology_extractions, f, indent=2)
    print(f"✅ Saved updated ontology to {updated_ontology_file_path}")

    return ontology_extractions, title_map, abstract_map, updated_ontology_file_path


def load_or_initialize_ontology(path: str) -> dict:
    """
    Load the ontology JSON file from the given path.
    If the file does not exist or fails to load, use a predefined fallback ontology.

    Args:
        path (str): The file path to the ontology JSON file.

    Returns:
        dict: The ontology data as a dictionary, either loaded from file or the fallback version.
    """
    if not os.path.exists(path):
        print("\u26A0\uFE0F 'ontology.json' not found.")
        return
    return json.load(open(path))


#### to be deleted
def fallback_ontology() -> dict:
    """
    Provide a predefined fallback ontology as a nested dictionary.

    This ontology serves as a default set of categories and keywords related to
    environmental conditions, processes, reactions, elemental compositions,
    materials, performance metrics, and applications, typically used when
    a user-provided ontology file is missing or cannot be loaded.

    Args:
        None.

    Returns:
        dict: A dictionary representing the ontology with hierarchical
              categories as keys and lists of related keywords/phrases as values.
    """
    return {
        "Environment": {
            "Acidic": ["pH < 7", "pH<7", "pH<", "pH 1", "pH=1", "acidic"],
            "Alkaline": ["pH > 7", "pH>7", "pH>", "pH=14", "alkaline", "basic"],
            "Neutral": ["pH 7", "neutral pH"],
            "Operating Temperature": ["temperature", "25\u00b0C", "80\u00b0C", "temperature-controlled"],
            "Pressure": ["1 atm", "10 atm", "high-pressure", "pressure"],
            "Gas-Phase Reactions": ["gas-phase", "gas reaction"],
            "Non-aqueous Solvents": ["non-aqueous", "organic solvent", "ethanol"]
        },
        "Process": {
            "Electrochemical Energy Conversion": ["electrochemical cell", "redox energy", "electrochemical process"],
            "Water Electrolysis": ["electrolysis of water", "splitting H2O", "hydrogen from water"],
            "Photoelectrochemical Water Splitting": ["photoelectrochemical", "PEC water splitting", "solar water splitting"],
            "Photocatalysis": ["photocatalyst", "UV-activated", "light-driven catalysis"],
            "Thermochemical Water Splitting": ["high-temperature splitting", "thermal decomposition of water"],
            "Electrocatalysis": ["electrocatalyst", "electrocatalytic", "electrode reaction enhancement"],
            "Fuel Cells": ["fuel cell", "proton exchange membrane", "PEM", "electricity generation from fuel"]
        },
        "Reaction": {
            "Hydrogen Evolution Reaction (HER)": ["hydrogen evolution", "HER", "H2 generation", "reduction of protons"],
            "Oxygen Evolution Reaction (OER)": ["oxygen evolution", "OER", "oxygen generation", "water oxidation"],
            "Proton-Coupled Electron Transfer (PCET)": ["PCET", "proton-electron", "concerted proton electron transfer"]
        },
        "Elemental Composition": {
            "Iron": ["Fe", "iron-based catalyst", "Fe3+", "Fe2+"],
            "Nickel": ["Ni", "nickel foam", "nickel oxide"],
            "Cobalt": ["Co", "Co3O4", "cobalt-doped"],
            "Manganese": ["Mn", "manganese oxide"],
            "Molybdenum": ["Mo", "MoS2", "molybdenum disulfide"],
            "Transition Metals": ["transition metal", "d-block metal", "first-row metals"],
            "Earth-Abundant Elements": ["earth-abundant", "non-precious", "low-cost metals"],
            "Metal-Free Catalysts": ["carbon-based", "N-doped graphene", "non-metal"],
            "Dopants": ["doped with P", "S-doping", "boron doping"]
        },
        "Material": {
            "Metal Oxides": ["Fe2O3", "NiO", "oxide layer", "metal oxide"],
            "Metal Phosphides": ["Ni2P", "FeP", "metal phosphide catalyst"],
            "Sulfides, Nitrides, Carbides": ["MoS2", "nitrides", "carbides", "WS2"],
            "Core–Shell Nanostructures": ["core-shell", "encapsulated nanoparticles"],
            "2D Materials": ["graphene", "MoS2", "nanosheets"],
            "Porous Structures, Nanorods, Nanowires": ["nanorods", "nanowires", "porous", "mesoporous", "nanoporous"]
        },
        "Performance & Stability": {
            "Overpotential": ["overpotential of", "η = ", "potential above thermodynamic"],
            "Tafel Slope": ["Tafel slope", "mV dec⁻¹", "Tafel plot"],
            "Exchange Current Density": ["exchange current density", "j0", "baseline current"],
            "Stability": ["chronoamperometry", "cycling stability", "durability test", "long-term performance"]
        },
        "Application": {
            "Green Hydrogen Economy": ["green hydrogen", "H2 economy", "renewable hydrogen"],
            "Hydrogen Production": ["H2 production", "hydrogen generation"],
            "Renewable Energy Integration": ["solar + water splitting", "renewable system", "wind-driven electrolysis"],
            "PEM Electrolyzer": ["proton exchange membrane", "PEM electrolyzer", "membrane cell"],
            "Lithium-Ion Battery": ["LIB", "lithium battery", "anode", "cathode"]
        }
    }


def load_paper_data(max_paper: int, paper_query_file_path) -> tuple[dict, dict, dict]:
    """
    Retrieve and organize research paper data up to a specified maximum number.

    Args:
        max_paper (int): The maximum number of unique papers to fetch.

    Returns:
        tuple:
            - text_data (dict): Maps a sanitized paper ID to its abstract text (empty string if missing).
            - title_map (dict): Maps the sanitized paper ID to the paper's title (empty string if missing).
            - abstract_map (dict): Maps the sanitized paper ID to the paper's abstract (empty string if missing).

    Description:
        This function fetches unique papers from a scholarly source, sanitizes their titles
        to create safe IDs, and organizes the abstracts and titles into dictionaries keyed
        by these safe IDs. If a paper's title or abstract is missing, an empty string is used instead.
    """
    text_data, title_map, abstract_map = {}, {}, {}
    papers = get_unique_papers_from_sch(max_paper, paper_query_file_path)
    for paper in papers:
        safe_id = safe_title(paper['title'] or paper['paper_id'])
        text_data[safe_id] = paper['abstract'] or ""
        title_map[safe_id] = paper['title'] or ""
        abstract_map[safe_id] = paper['abstract'] or ""
    return text_data, title_map, abstract_map


def extract_keywords(ontology: dict, text_data: dict, api_keys: list) -> dict:
    """
    Extract relevant keywords from paper abstracts for each ontology layer using the OpenAI API.

    Args:
        ontology (dict): The ontology structure mapping layers to categories and associated keywords.
        text_data (dict): A mapping of paper IDs to their abstract texts.
        api_keys (list): A list of OpenAI API keys to use for querying the model.

    Returns:
        dict: A nested dictionary (defaultdict of defaultdict of lists) structured as:
            {
                layer: {
                    category: [
                        {
                            "keywords": list of unique keywords including ontology keywords and matched keyword,
                            "meta_data": metadata extracted by the model for the keyword,
                            "paper_id": ID of the paper where the keyword was found
                        }, ...
                    ], ...
                }, ...
            }

    Description:
        For each paper abstract, this function iterates over each ontology layer and its categories,
        queries the OpenAI model to extract matched keywords and metadata,
        and collects these extractions in a structured dictionary.
        It skips any categories returned by the model that are not defined in the ontology.
        Progress and warnings are printed to the console during processing.
    """
    
    results_dict = defaultdict(lambda: defaultdict(list))
    for paper_id, abstract in text_data.items():
        print(f"\U0001F50D Processing paper: {paper_id}")
        for layer, categories in ontology.items():
            results = call_openai_model(layer, abstract, categories, api_keys)
            for res in results:
                cat = res["matched_category"]
                if cat not in ontology[layer]:
                    print(f"\u26A0\uFE0F Skipping unknown category '{cat}' under layer '{layer}'")
                    continue
                keywords = list(set(ontology[layer][cat] + [res["keyword"]]))
                results_dict[layer][cat].append({
                    "keywords": keywords,
                    "meta_data": res["meta_data"],
                    "paper_id": paper_id
                })
            print(f"\u2705 Total items extracted for paper {paper_id}, layer '{layer}': {len(results)}")
    return results_dict


def parse_all_metadata(ontology_extractions: dict) -> None:
    """
    Parse metadata for each keyword extraction within the ontology extractions.

    Args:
        ontology_extractions (dict): Nested dictionary of extracted keywords and metadata,
            structured as {layer: {category: [items]}}, where each item contains:
            - "meta_data" (str): Raw metadata string extracted from the text.
            - other keys such as "keywords" and "paper_id".

    Returns:
        None: This function modifies the input dictionary in-place by adding parsed metadata fields.

    Description:
        Iterates over all extracted keyword items in the ontology_extractions dictionary,
        parsing the raw "meta_data" string with the `parse_meta_data` function.
        It adds a new "parsed_meta" field to each item with the parsed results.
        Additionally, if the raw metadata contains keywords like "overpotential" or "tafel",
        it extracts and stores specific parsed values under "overpotential" and "tafel_slope" keys respectively.
    """
    for layer, cats in ontology_extractions.items():
        for cat, items in cats.items():
            for item in items:
                item["parsed_meta"] = parse_meta_data(item["meta_data"])
                meta_lower = item["meta_data"].lower()
                for parsed in item["parsed_meta"]:
                    if "overpotential" in meta_lower:
                        item["overpotential"] = parsed
                    if "tafel" in meta_lower:
                        item["tafel_slope"] = parsed



def parse_meta_data(meta_str: str) -> list[dict]:
    """
    Parse numerical values and their units from a metadata string.

    Args:
        meta_str (str): A raw metadata string potentially containing numbers with units,
            e.g., "10 cm⁻²", "−5.2 µA", "3.1e-4 mV".

    Returns:
        list of dict: A list of dictionaries, each containing:
            - "value" (float): The numerical value parsed from the string.
            - "unit" (str): The associated unit string.

    Description:
        The function first normalizes certain Unicode characters and formatting inconsistencies,
        such as replacing "cm⁻²" with "cm^-2", and normalizing minus signs and micro symbols.
        Then it uses a regular expression to find all occurrences of numeric values (including
        scientific notation) followed by unit strings (including Greek letters, degree symbols,
        and common unit characters). It returns all matches as a list of dictionaries with
        parsed numeric values and their units.
    """
    # Normalize formatting
    meta_str = meta_str.replace("cm\u22122", "cm^-2")
    meta_str = meta_str.replace("−", "-").replace("µ", "u").replace("μ", "u")

    # Pattern for numbers + scientific units
    pattern = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Zμµ°²^/%\-]+)"
    return [{"value": float(val), "unit": unit} for val, unit in re.findall(pattern, meta_str)]

# ✅ Save Semantic Scholar metadata to Excel
def export_semantic_scholar_metadata_to_excel(paper_list, output_file="semantic_scholar_papers.xlsx"):
    if not paper_list:
        print("⚠️ No papers to export.")
        return

    df = pd.DataFrame(paper_list)

    # Show preview
    print("📄 Exporting the following columns:", list(df.columns))
    print(f"✅ Total papers: {len(df)}")

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"✅ Saved to '{output_file}'")




def build_knowledge_graph_from_ontology_json(
    ontology_path: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pass: str
) -> None:
    """
    Reads a JSON file containing an extracted ontology and builds a knowledge graph in Neo4j.

    Parameters
    ----------
    ontology_path : str
        Path to the input JSON file containing the extracted ontology structure.
    neo4j_uri : str
        Connection URI for the Neo4j instance (e.g., bolt://localhost:7687 or neo4j+s://...).
    neo4j_user : str
        Username for Neo4j authentication.
    neo4j_pass : str
        Password for Neo4j authentication.

    Description
    -----------
    This function processes an ontology file organized by layers, categories, and associated
    keywords with metadata. It populates a Neo4j knowledge graph with the following nodes:
    
    - Layer
    - Category
    - Keyword
    - MetaData
    - Paper
    
    It also creates relationships between them:
    
    - Layer → HAS_CATEGORY → Category
    - Category → HAS_KEYWORD → Keyword
    - Keyword → HAS_METADATA → MetaData
    - Keyword → MENTIONS → Paper

    Notes
    -----
    - Duplicates are prevented using in-memory tracking sets.
    - Keywords are uniquely identified by their layer, category, and text.
    - Metadata is connected only if available for each keyword.

    Returns
    -------
    None
    """

    # Load extracted ontology from JSON file (updated Json file)
    with open(ontology_path, "r") as f:
        ontology_extractions = json.load(f)

    # Connect to Neo4j using provided credentials
    graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_pass))

    # Track uniqueness to avoid duplicate nodes and relationships
    unique_keywords = defaultdict(set)
    unique_categories = set()
    unique_layers = set()
    paper_keywords = defaultdict(set)  # Store keyword associations per paper

    # -----------------------------------------------------
    # Step 1: Create Layer, Category, Keyword, MetaData nodes
    # -----------------------------------------------------
    for layer, categories in ontology_extractions.items():
        # Create or reuse Layer node
        if layer not in unique_layers:
            layer_node = Node("Layer", name=layer)
            graph.merge(layer_node, "Layer", "name")
            unique_layers.add(layer)
        else:
            layer_node = graph.nodes.match("Layer", name=layer).first()

        for cat, items in categories.items():
            cat_key = f"{layer}|{cat}"

            # Create or reuse Category node and link it to Layer
            if cat_key not in unique_categories:
                cat_node = Node("Category", name=cat, key=cat_key)
                graph.merge(cat_node, "Category", "key")
                graph.merge(Relationship(layer_node, "HAS_CATEGORY", cat_node))
                unique_categories.add(cat_key)
            else:
                cat_node = graph.nodes.match("Category", key=cat_key).first()

            # Process each keyword item under this category
            for item in items:
                paper_id = item.get("paper_id", "unknown")
                for kw in item["keywords"]:
                    kw_key = f"{layer}|{cat}|{kw}"

                    # Skip if keyword already processed
                    if kw_key in unique_keywords[(layer, cat)]:
                        continue
                    unique_keywords[(layer, cat)].add(kw_key)

                    # Create Keyword node and link to Category
                    kw_node = Node("Keyword", name=kw, key=kw_key)
                    graph.merge(kw_node, "Keyword", "key")
                    graph.merge(Relationship(cat_node, "HAS_KEYWORD", kw_node))

                    # Create and link MetaData nodes to the keyword
                    for meta in item.get("parsed_meta", []):
                        meta_node = Node(
                            "MetaData",
                            value=meta["value"],
                            unit=meta["unit"],
                            type=cat
                        )
                        graph.create(meta_node)
                        graph.create(Relationship(kw_node, "HAS_METADATA", meta_node))

                    # Track keyword association for this paper
                    paper_keywords[paper_id].add(kw_key)

    print("✅ Layer, Category, Keyword, and MetaData nodes created")

    # -----------------------------------------------------
    # Step 2: Create Paper nodes and MENTIONS relationships
    # -----------------------------------------------------
    for paper_id, kw_keys in paper_keywords.items():
        # Create or reuse Paper node
        paper_node = Node("Paper", id=paper_id)
        graph.merge(paper_node, "Paper", "id")

        # Link each associated Keyword to the Paper
        for kw_key in kw_keys:
            kw_node = graph.nodes.match("Keyword", key=kw_key).first()
            if kw_node:
                rel = Relationship(kw_node, "MENTIONS", paper_node)
                graph.merge(rel)

    print("✅ Paper nodes and MENTIONS relationships created")
    print("🎉 Knowledge graph construction complete.")


def classify_query_with_fallback(user_query: str, ontology: dict, client, kw_lookup: dict) -> list:
    """
    Classify a user query into ontology categories using OpenAI, with fallback logic.

    Parameters:
    - user_query (str): The natural language query input from the user.
    - ontology (dict): A nested dictionary representing the ontology structure.
    - client: An OpenAI API client instance for calling GPT-4o.
    - kw_lookup (dict): A dictionary mapping keywords to (layer, category) for fallback classification.

    Returns:
    - categories (list): A list of tuples (layer, category, keyword) representing matched ontology classifications.
    """

    # Tokenize the query: extract all alphanumeric tokens longer than one character
    tokens = [t for t in re.findall(r"\w+", user_query.lower()) if len(t) > 1]

    # Build a simplified summary of the ontology for inclusion in the prompt
    ontology_summary = {layer: list(cats.keys()) for layer, cats in ontology.items()}

    # Create the prompt to send to OpenAI's chat model
    prompt = f'''
        You are a structured classification AI for materials science.
        Ontology:
        {json.dumps(ontology_summary, indent=2)}

        Given the user query: "{user_query}"

        Extract each meaningful keyword (ignore stopwords like 'and', 'for', etc).
        Assign each keyword to the most relevant category and layer from the ontology.
        Return only the results in valid JSON format:

        [
        {{
            "keyword": "cheap",
            "category": "Earth-Abundant Elements",
            "layer": "Elemental Composition"
        }},
        ...
        ]
    '''

    # Call OpenAI's GPT model to classify keywords
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    # Extract and clean the model's output
    result = response.choices[0].message.content.strip()
    result = result.replace("```json", "").replace("```", "")

    # Parse the model's JSON output
    try:
        model_output = json.loads(result)
    except json.JSONDecodeError:
        print("⚠️ OpenAI output couldn't be parsed. Raw:\n", result)
        model_output = []

    categories = []        # Final list of classified keywords
    found_set = set()      # Tracks (layer, category) to avoid duplicates
    model_keywords = set() # Tracks which keywords were already classified by the model

    # Process the model's output
    for item in model_output:
        kw = item["keyword"].lower()
        model_keywords.add(kw)
        if item["layer"] != "Unknown" and item["category"] != "Unknown":
            if (item["layer"], item["category"]) not in found_set:
                categories.append((item["layer"], item["category"], kw))
                found_set.add((item["layer"], item["category"]))
                print(f"✅ OpenAI: '{kw}' → {item['layer']} / {item['category']}")

    # Fallback for unclassified keywords using the local lookup
    for tok in tokens:
        if tok not in model_keywords:
            if tok in kw_lookup:
                layer, cat = kw_lookup[tok]
                if (layer, cat) not in found_set:
                    categories.append((layer, cat, tok))
                    found_set.add((layer, cat))
                    print(f"🛟 Fallback: '{tok}' → {layer} / {cat}")
            else:
                print(f"❌ No match for: '{tok}'")

    return categories



def summarize_filtered_papers_with_opposites(
    filtered_out: Dict[str, Dict[str, Any]],
    query_keywords: List[str],
    opposites: Dict[str, List[str]],
    paper_keyword_frequencies: Dict[str, Dict[str, int]],
    title_map: Dict[str, str],
    abstract_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Summarizes why certain papers were filtered out based on presence of opposing keywords.

    Parameters:
    - filtered_out (dict): Dictionary of paper IDs mapped to filtering metadata (e.g., threshold).
    - query_keywords (list): List of relevant query keywords used for matching.
    - opposites (dict): Dictionary mapping each query keyword to its list of opposing keywords.
    - paper_keyword_frequencies (dict): Not used in the current implementation but may be needed for extensions.
    - title_map (dict): Maps paper IDs to their title texts.
    - abstract_map (dict): Maps paper IDs to their abstract texts.

    Returns:
    - pd.DataFrame: A summary table showing keyword matches in title/abstract and reasons for filtering.
    """

    rows = []

    # Loop through each filtered-out paper
    for pid, info in filtered_out.items():
        title_text = (title_map.get(pid, "") or "").lower()
        abstract_text = (abstract_map.get(pid, "") or "").lower()

        # Evaluate each keyword from the user query
        for qk in query_keywords:
            matched_opp_keywords = []

            # --- Title Matching ---
            # Count exact matches of the query keyword in title
            title_rel = len(re.findall(rf'\b{re.escape(qk)}\b', title_text))
            title_opp = 0

            # Count opposing keywords in title
            for opp in opposites.get(qk, []):
                count = len(re.findall(rf'\b{re.escape(opp)}\b', title_text))
                title_opp += count
                if count > 0:
                    matched_opp_keywords.append(opp)

            # --- Abstract Matching ---
            # Count query keyword matches in abstract
            abs_rel = len(re.findall(rf'\b{re.escape(qk)}\b', abstract_text))
            abs_opp = 0

            # Count opposing keywords in abstract
            for opp in opposites.get(qk, []):
                count = len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text))
                abs_opp += count
                if count > 0 and opp not in matched_opp_keywords:
                    matched_opp_keywords.append(opp)

            # Total counts
            total_rel = title_rel + abs_rel
            total_opp = title_opp + abs_opp

            # Compute ratio of opposing to relevant keywords
            ratio = round(total_opp / total_rel, 2) if total_rel else float('inf')

            # Determine filtering status
            status = "Filtered" if ratio > info["threshold"] else "Kept"

            # Collect results
            rows.append({
                "Paper ID": pid,
                "Query Keyword": qk,
                "Title Relevant Count": title_rel,
                "Title Opposing Count": title_opp,
                "Abstract Relevant Count": abs_rel,
                "Abstract Opposing Count": abs_opp,
                "Total Relevant Count": total_rel,
                "Total Opposing Count": total_opp,
                "Matched Opposing Keywords": ", ".join(sorted(set(matched_opp_keywords))),
                "Ratio": ratio,
                "Status": status
            })

    return pd.DataFrame(rows)



def get_opposites_via_llm(keywords: List[str], client: Any) -> Dict[str, List[str]]:
    """
    Uses an LLM (e.g., OpenAI GPT-4o) to generate opposing or contradictory keywords
    for a given list of scientific keywords, specifically in the context of
    electrocatalysis, water splitting, or electrochemical environments.

    Parameters:
    - keywords (List[str]): List of scientific keywords (e.g., "acidic", "HER").
    - client (Any): OpenAI client instance used to send the prompt.

    Returns:
    - Dict[str, List[str]]: A dictionary where each keyword maps to a list of opposing terms.
    """
    
    # Construct the prompt for the language model
    prompt = f"""
        You are a materials science assistant.

        For each of the following scientific keywords, return a list of their opposite or contradictory concepts 
        in the context of electrocatalysis, water splitting, or electrochemical environments.

        Keywords: {keywords}

        Respond only in JSON format like this:
        {{
          "acidic": ["alkaline", "basic"],
          "HER": ["OER", "oxygen evolution"],
          ...
        }}
    """

    # Send prompt to the LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Zero temperature for deterministic output
    )

    # Clean the response content
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    # Attempt to parse the response as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Could not parse opposite keyword JSON. Raw output:\n", content)
        return {}


def get_synonyms_via_llm(keywords: List[str], client: Any) -> Dict[str, List[str]]:
    """
    Uses an LLM (e.g., OpenAI GPT-4o) to generate synonyms or semantically similar terms
    for a given list of scientific keywords. This is specific to language used in 
    electrocatalysis, water splitting, or electrochemical contexts.

    Parameters:
    - keywords (List[str]): A list of scientific terms (e.g., "HER", "acidic").
    - client (Any): The OpenAI API client used to submit the query.

    Returns:
    - Dict[str, List[str]]: A dictionary mapping each input keyword to a list of synonyms.
    """

    # Create a clear and structured prompt for the model
    prompt = f"""
        You are a materials science assistant.

        For each of the following scientific keywords, return a list of their synonyms or semantically 
        similar expressions used in electrocatalysis, water splitting, or electrochemistry papers.

        Keywords: {keywords}

        Respond only in JSON format like this:
        {{
          "acidic": ["low pH", "acid media"],
          "HER": ["hydrogen evolution", "H2 generation"],
          ...
        }}
    """

    # Make the API call to the OpenAI chat completion model
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # Deterministic output
    )

    # Clean up the raw model response
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    # Attempt to parse the response as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Could not parse synonym keyword JSON. Raw output:\n", content)
        return {}


def rank_by_pair_overlap_filtered(
    per_cat_hits: Dict[Tuple[str, str], Dict[str, float]],
    ranked_paper_ids: Set[str]
) -> Tuple[List[Tuple[str, Set[str]]], int]:
    """
    Ranks papers based on how many unique category-pair overlaps they appear in,
    considering only a filtered set of paper IDs.

    Parameters:
    - per_cat_hits: A dictionary where each key is a (layer, category) tuple and
      the value is another dict mapping paper IDs to their relevance scores.
    - ranked_paper_ids: A set of paper IDs that have passed some prior filtering.

    Returns:
    - ranked_by_pair_overlap: A list of tuples (paper ID, set of category pair labels),
      sorted by the number of unique category-pair overlaps in descending order.
    - total_possible_pairs: The total number of unique category-category pairs.
    """

    pair_paper_map = defaultdict(set)  # Maps each paper ID to the set of category pairs it overlaps in
    category_pairs = list(combinations(per_cat_hits.keys(), 2))  # All unique pairs of (layer, category)
    total_possible_pairs = len(category_pairs)

    for (cat1, cat2) in category_pairs:
        papers1 = set(per_cat_hits[cat1].keys())  # All paper IDs under category 1
        papers2 = set(per_cat_hits[cat2].keys())  # All paper IDs under category 2
        shared_papers = papers1 & papers2         # Papers common to both categories
        filtered_shared = shared_papers & ranked_paper_ids  # Only consider filtered/ranked paper IDs

        pair_label = f"{cat1[1]} ↔ {cat2[1]}"  # Use only category names for display

        for pid in filtered_shared:
            pair_paper_map[pid].add(pair_label)  # Record that this paper overlaps this pair

    # Rank papers by number of unique overlapping category pairs (descending)
    ranked_by_pair_overlap = sorted(pair_paper_map.items(), key=lambda x: len(x[1]), reverse=True)

    return ranked_by_pair_overlap, total_possible_pairs



def is_dominated_by_opposites(
    paper_id: str,
    title_map: Dict[str, str],
    abstract_map: Dict[str, str],
    query_keywords: List[str],
    expanded_keywords: Dict[str, List[str]],
    opposites: Dict[str, List[str]],
    filtered_out: Dict[str, Dict]
) -> bool:
    """
    Determines whether a paper should be filtered out based on whether opposing keywords
    dominate relevant ones in the title and abstract.

    Parameters:
    - paper_id: Unique identifier of the paper.
    - title_map: Mapping from paper ID to title text.
    - abstract_map: Mapping from paper ID to abstract text.
    - query_keywords: List of original query keywords.
    - expanded_keywords: Dictionary mapping each keyword to its expanded synonym list.
    - opposites: Dictionary mapping each keyword to its list of opposing terms.
    - filtered_out: Dictionary to record reasons for filtering papers.

    Returns:
    - True if the paper is filtered out (dominated by opposites), otherwise False.
    """

    # Normalize and prepare title and abstract (excluding references section)
    title_text = (title_map.get(paper_id, "") or "").lower()
    full_abstract = (abstract_map.get(paper_id, "") or "").lower()
    abstract_text = re.split(r'(references|refs?\.?:)', full_abstract)[0]

    title_flags = {}
    title_over_threshold = 0

    # Analyze keyword relevance and opposition in title
    for qk in query_keywords:
        relevant_terms = expanded_keywords.get(qk, [qk])
        title_rel = sum(len(re.findall(rf'\b{re.escape(term)}\b', title_text)) for term in relevant_terms)
        title_opp = sum(len(re.findall(rf'\b{re.escape(opp)}\b', title_text)) for opp in opposites.get(qk, []))

        if title_rel == 0 and title_opp == 0:
            title_flags[qk] = "missing"
        else:
            ratio = float('inf') if title_rel == 0 else title_opp / title_rel
            if ratio > 1:
                title_flags[qk] = "fail"
                title_over_threshold += 1
            elif ratio in [0, 1]:
                title_flags[qk] = "pass"
            else:
                title_flags[qk] = "other"

    # 🔴 Case 1: Low relevance - Opposing terms dominate in title
    if "fail" in title_flags.values():
        filtered_out[paper_id] = {
            "reason": "title_fail",
            "title_flags": title_flags,
            "title_exceeded": title_over_threshold,
            "threshold": 1.0,
        }
        return True

    # 🟡 Case 2: Moderate relevance - Mixed signal in title, check abstract
    if "pass" in title_flags.values() and "missing" in title_flags.values():
        abstract_ratios = []

        for qk in query_keywords:
            relevant_terms = expanded_keywords.get(qk, [qk])
            abs_rel = sum(len(re.findall(rf'\b{re.escape(term)}\b', abstract_text)) for term in relevant_terms)
            abs_opp = sum(len(re.findall(rf'\b{re.escape(opp)}\b', abstract_text)) for opp in opposites.get(qk, []))
            ratio = float('inf') if abs_rel == 0 else abs_opp / abs_rel
            abstract_ratios.append(ratio)

        if all(r <= 1.5 for r in abstract_ratios):
            return False  # Considered high relevance despite title mix
        else:
            filtered_out[paper_id] = {
                "reason": "moderate_abstract_check",
                "abstract_ratios": abstract_ratios,
                "title_exceeded": 0,
                "threshold": 1.5,
            }
            return False  # Not excluded, but marked for review

    # 🟢 Case 3: High relevance - All keywords pass in title
    if all(flag == "pass" for flag in title_flags.values()):
        return False

    # Default fallback: Keep the paper
    return False

#### user_query = keywords_query_list (str)
def find_common_papers(
    user_query: str,
    ontology_extractions: dict,
    category_to_papers: dict,
    paper_keyword_map: dict,
    title_map: dict,
    client,
    kw_lookup: dict,
    abstract_map: dict,
    output_dir,
    threshold: float = 1.5
) -> tuple:
    """
    Main pipeline to process a user query, classify it into ontology categories,
    filter papers dominated by opposite keywords, rank relevant papers, and
    analyze overlaps between categories.

    Args:
        user_query (str): The search query input by the user.
        ontology_extractions (dict): Ontology categories extracted for classification.
        category_to_papers (dict): Mapping from (layer, category) to papers and their counts.
        paper_keyword_map (dict): Keywords per paper, per category.
        title_map (dict): Paper titles keyed by paper ID.
        client: Client instance for LLM calls.
        kw_lookup (dict): Keyword lookup dictionary.
        abstract_map (dict): Paper abstracts keyed by paper ID.
        threshold (float): Threshold for filtering opposite keyword dominance (default 1.5).

    Returns:
        tuple: 
            - ranked (list): List of tuples (paper_id, score) for ranked papers.
            - low_sorted (list): List of filtered low relevance papers sorted by threshold exceed count.
    """
    #@Sana : the code didnt use threshold. what is that?


    print(f"🔍 User query: {user_query}")

    # Step 1: Classify the user query to ontology categories with fallback
    categories = classify_query_with_fallback(user_query, ontology_extractions, client, kw_lookup)
    if not categories:
        print("⚠️ No valid categories found.")
        return

    print(f"\n✅ Categories Used: {len(categories)}")
    for i, (layer, cat, kw) in enumerate(categories, 1):
        print(f"  {i}. '{kw}' → {layer} / {cat}")

    # Extract and lowercase keywords for consistent matching
    query_keywords = [kw.lower() for _, _, kw in categories]

    # Step 2: Fetch opposites and synonyms for keywords via LLM
    opposites = get_opposites_via_llm(query_keywords, client)
    synonyms = get_synonyms_via_llm(query_keywords, client)

    # Build expanded keywords dictionary including synonyms
    expanded_keywords = {kw: [kw] + synonyms.get(kw, []) for kw in query_keywords}

    print(f"\n🧪 Opposites used for filtering:")
    for qk in query_keywords:
        print(f"  - '{qk}': {opposites.get(qk, [])}")

    # Step 3: Aggregate paper hits and keyword frequencies per category
    per_cat_hits = {}
    total_scores = defaultdict(int)
    paper_keyword_frequencies = defaultdict(lambda: defaultdict(int))
    filtered_out = {}

    for (layer, cat), hits in category_to_papers.items():
        if (layer, cat) not in [(l, c) for l, c, _ in categories]:
            continue

        per_cat_hits[(layer, cat)] = hits

        for pid, count in hits.items():
            total_scores[pid] += count
            for kw in paper_keyword_map[(layer, cat)][pid]:
                paper_keyword_frequencies[pid][kw] += 1

    # Step 4: Filter out papers dominated by opposite keywords
    all_paper_ids = list(title_map.keys())
    ranked = []

    for pid in all_paper_ids:
        score = total_scores.get(pid, 0)
        if not is_dominated_by_opposites(
            pid, title_map, abstract_map, query_keywords, expanded_keywords, opposites, filtered_out
        ):
            ranked.append((pid, score))

    ranked_paper_ids = {pid for pid, _ in ranked}

    # Step 5: Display report for filtered out papers, if any
    if filtered_out:
        summary_df = summarize_filtered_papers_with_opposites(
            filtered_out, query_keywords, opposites, paper_keyword_frequencies, title_map, abstract_map
        )
        try:
            from IPython.display import display
            display(summary_df)
        except ImportError:
            pass

        summary_df.to_csv(f"{output_dir}/filtered_paper_breakdown.csv", index=False)

        print(f"\n🚫 Filtered out {len(filtered_out)} papers due to dominance of opposite keywords.")

    # Step 6: Separate and sort low relevance papers (title fail)
    low_relevance = [(pid, info.get("title_exceeded", 0)) for pid, info in filtered_out.items() if info["reason"] == "title_fail"]
    low_sorted = sorted(low_relevance, key=lambda x: x[1])

    # Step 7: Print ranked paper results
    print("\n📚 Ranked Papers:")
    print("\n🟢🟡 High & Moderate Relevance:")
    for i, (pid, score) in enumerate(sorted(ranked, key=lambda x: -x[1]), 1):
        print(f"  {i}. {pid} (mentions={score})")

    print("\n🔴 Low Relevance (sorted by number of over-threshold keywords):")
    for i, (pid, count) in enumerate(low_sorted, 1):
        print(f"  {i}. {pid} (keywords over threshold: {count})")

    # Step 8: Filter hits to only ranked papers for overlap analysis
    filtered_hits = {
        k: {pid: v for pid, v in d.items() if pid in ranked_paper_ids}
        for k, d in per_cat_hits.items()
    }

    if len(per_cat_hits) < 2:
        print("\n⚠️ Overlap analysis needs 2+ categories.")
        return ranked, low_sorted

    # Step 9: Rank papers by number of category pairs they appear in
    ranked_by_pair_overlap, total_possible_pairs = rank_by_pair_overlap_filtered(filtered_hits, ranked_paper_ids)

    print(f"\n🏆 Full Ranking by Number of Category Pairs Shared (out of {total_possible_pairs} pairs):")
    for i, (pid, pair_set) in enumerate(ranked_by_pair_overlap, 1):
        pair_count = len(pair_set)
        print(f"  {i}. {pid} → shared in {pair_count}/{total_possible_pairs} pair(s)")
        print(f"     ⤷ Pairs: {sorted(pair_set)}")

    if not ranked_by_pair_overlap:
        print("⚠️ No multi-category-pair overlaps found.")

    # Export overlap ranking to CSV
    df_pairs = pd.DataFrame([
        {
            "Paper ID": pid,
            "Pair Count": len(pair_set),
            "Shared Pairs": ", ".join(sorted(pair_set))
        }
        for pid, pair_set in ranked_by_pair_overlap
    ])

    df_pairs_file_path = f"{output_dir}/shared_pair_ranked_papers.csv"

    df_pairs.to_csv(df_pairs_file_path, index=False)
    print("✅ Exported shared papers to 'shared_pair_ranked_papers.csv'")

    # Step 10: Show category overlap statistics
    print("\n🔄 Overlaps Between Categories:")
    for (l1, c1), (l2, c2) in combinations(per_cat_hits.keys(), 2):
        pids1 = set(paper_keyword_map[(l1, c1)].keys())
        pids2 = set(paper_keyword_map[(l2, c2)].keys())
        shared = pids1 & pids2
        filtered_shared = shared & ranked_paper_ids

        total_mentions = sum(
            len(paper_keyword_map[(l1, c1)][pid]) + len(paper_keyword_map[(l2, c2)][pid])
            for pid in filtered_shared
        )

        print(f"  • '{c1}' ↔ '{c2}': {len(filtered_shared)} shared papers, {total_mentions} keyword mentions in common")

    return ranked, low_sorted, df_pairs_file_path


def rank_shared_papers_by_pairs_and_mentions(
    ranked: list,
    low_sorted: list,
    pair_file: str 
) -> pd.DataFrame:
    """
    Ranks shared papers by pair count and keyword relevance.

    This function combines the paper relevance scores (from keyword matching) with pair frequency data,
    then ranks them based on both metrics.

    Args:
        ranked (list): List of tuples (paper_id, keyword_score) for highly relevant papers.
        low_sorted (list): List of tuples (paper_id, count) for less relevant papers.
        pair_file (str): Path to CSV file containing 'Paper ID' and 'Pair Count' columns.

    Returns:
        pd.DataFrame: Sorted DataFrame with relevance, keyword scores, and pair counts.
    """

    # Step 1: Load paper pair data
    try:
        df_pairs = pd.read_csv(pair_file)
    except FileNotFoundError:
        print(f"❌ File not found: {pair_file}")
        return pd.DataFrame()

    # Step 2: Build a mapping of paper_id to (relevance_level, score)
    relevance_scores = {paper_id: ("high", score) for paper_id, score in ranked}
    for paper_id, _ in low_sorted:
        if paper_id not in relevance_scores:
            relevance_scores[paper_id] = ("low", 0)

    # Step 3: Annotate DataFrame with relevance level and keyword match score
    df_pairs["Relevance Level"] = df_pairs["Paper ID"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[0])
    df_pairs["Relevant Keyword Score"] = df_pairs["Paper ID"].map(lambda x: relevance_scores.get(x, ("unknown", 0))[1])

    # Step 4: Sort by pair count (descending), then by keyword score (descending)
    df_pairs = df_pairs.sort_values(
        by=["Pair Count", "Relevant Keyword Score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Step 5: Print ranked list
    print("\n🏆 Shared Papers Ranked by Pair Count → Mentions:")
    for _, row in df_pairs.iterrows():
        print(f"{row['Paper ID']} | Pairs: {row['Pair Count']} | Mentions: {row['Relevant Keyword Score']} | Relevance: {row['Relevance Level']}")

    # Step 6: Save subsets by relevance level
    for level in ["high", "low", "unknown"]:
        subset = df_pairs[df_pairs["Relevance Level"] == level]
        if not subset.empty:
            filename = f"shared_ranked_by_pairs_then_mentions_{level}.csv"
            subset.to_csv(filename, index=False)
            print(f"✅ Saved: {filename}")

    return df_pairs

def ranking_papers(
    API_file_path,
    ontology_path,
    max_papers,
    paper_query_file_path,
    keyword_query_file_path,
    output_dir
) -> None:
    """
    Load ontology data, build mappings from ontology categories to papers and keywords,
    classify and rank papers relevant to a query, then rank shared papers by category pairs.

    Args:
        title_map (dict): Mapping of paper IDs to their titles.
        abstract_map (dict): Mapping of paper IDs to their abstracts.
        pair_file (str): Path to file containing category pair data for ranking overlaps.

    Returns:
        None: Prints output and saves ranked paper info, no explicit return.
    """

    # Step 1: Load ontology extraction data from JSON file (old code)
    # with open("updated_ontology.json", "r") as f:
    #     ontology_extractions = json.load(f)

    # new code 
    # Step 1: Load ontology extraction data
    ontology_extractions, title_map, abstract_map, updated_ontology_file_path = enrich_ontology_with_papers(ontology_path, API_file_path, max_papers, paper_query_file_path, output_dir)

    # Step 2: Initialize dictionaries to store mappings and counts
    category_to_papers = defaultdict(lambda: defaultdict(int))   # {(layer, category): {paper_id: count}}
    paper_keyword_map = defaultdict(lambda: defaultdict(set))    # {(layer, category): {paper_id: set(keywords)}}
    kw_lookup = {}  # Keyword to (layer, category) lookup for quick reference

    # Step 3: Populate mappings from ontology extraction data
    for layer, categories in ontology_extractions.items():
        for category, items in categories.items():
            for item in items:
                paper_id = item["paper_id"]
                for kw in item["keywords"]:
                    kw_lower = kw.lower()
                    kw_lookup[kw_lower] = (layer, category)
                    paper_keyword_map[(layer, category)][paper_id].add(kw_lower)
                    category_to_papers[(layer, category)][paper_id] += 1

    # Step 4: Initialize OpenAI client (make sure get_next_api_key() is implemented)
    API_keys = load_api_keys(API_file_path)
    client = OpenAI(api_key=get_next_api_key(API_keys))

    # Step 5: Run the main query processing and ranking pipeline
    # Here, "acidic HER water splitting" is the example search query , user_query="acidic HER water splitting"
    with open(keyword_query_file_path, "r") as file:
        user_query = file.read()
    ranked, low_sorted, pair_file_path = find_common_papers(
        user_query=user_query,
        ontology_extractions=ontology_extractions,
        category_to_papers=category_to_papers,
        paper_keyword_map=paper_keyword_map,
        title_map=title_map,
        client=client,
        kw_lookup=kw_lookup,
        abstract_map=abstract_map,
        output_dir = output_dir,
        threshold=1.5

    )

    # Step 6: Further rank shared papers based on pairs and mentions in the pair_file
    final_ranked_shared = rank_shared_papers_by_pairs_and_mentions(ranked, low_sorted, pair_file_path)

    # You can print or save final_ranked_shared as needed, for example:
    print("\n🎯 Final shared paper ranking:")
    for idx, (paper_id, score) in enumerate(final_ranked_shared, 1):
        print(f"{idx}. {paper_id}: {score}")
    
    return updated_ontology_file_path