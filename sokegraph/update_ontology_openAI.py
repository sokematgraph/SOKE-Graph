import os
from sokegraph.util.logger import LOG
import json
from collections import defaultdict
import re
from openai import OpenAI
from sokegraph.functions import get_next_api_key
from typing import List, Dict


def load_ontology(path: str) -> dict:
    """
    Load the ontology JSON file from the given path.
    If the file does not exist or fails to load, use a predefined fallback ontology.

    Args:
        path (str): The file path to the ontology JSON file.

    Returns:
        dict: The ontology data as a dictionary, either loaded from file or the fallback version.
    """
    if not os.path.exists(path):
        LOG.error("\u26A0\uFE0F 'ontology.json' not found.")
        #print("\u26A0\uFE0F 'ontology.json' not found.")
        return
    
    LOG.info("Loaded ontology file")
    return json.load(open(path))

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
        #print("📄 OpenAI returned:\n", model_output[:500])  # Preview first 500 characters

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
            LOG.info(f"✅ Extracted {len(extracted)} items via regex.")
            #print(f"✅ Extracted {len(extracted)} items via regex.")
            return extracted

    except Exception as e:
        print(f"❌ OpenAI API call failed: {e}")
        return []


def extract_keywords_openAI(ontology: dict, text_data: dict, api_keys: list) -> dict:
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
        LOG.info(f"\U0001F50D Processing paper: {paper_id}")
        #print(f"\U0001F50D Processing paper: {paper_id}")
        for layer, categories in ontology.items():
            results = call_openai_model(layer, abstract, categories, api_keys)
            for res in results:
                cat = res["matched_category"]
                if cat not in ontology[layer]:
                    LOG.info(f"\u26A0\uFE0F Skipping unknown category '{cat}' under layer '{layer}'")
                    #print(f"\u26A0\uFE0F Skipping unknown category '{cat}' under layer '{layer}'")
                    continue
                keywords = list(set(ontology[layer][cat] + [res["keyword"]]))
                results_dict[layer][cat].append({
                    "keywords": keywords,
                    "meta_data": res["meta_data"],
                    "paper_id": paper_id
                })
            LOG.info(f"\u2705 Total items extracted for paper {paper_id}, layer '{layer}': {len(results)}")
            #print(f"\u2705 Total items extracted for paper {paper_id}, layer '{layer}': {len(results)}")
    #import pdb
    #pdb.set_trace()

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
    
    #import pdb
    #pdb.set_trace()

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


def enrich_ontology_with_papers_openAI(ontology_path: str,
                                api_keys,
                                text_data,
                                output_dir):
    
    ontology = load_ontology(ontology_path)
    ontology_extractions = extract_keywords_openAI(ontology, text_data, api_keys)
    parse_all_metadata(ontology_extractions)
    updated_ontology_file_path = f"{output_dir}/updated_ontology.json"
    with open(updated_ontology_file_path, "w") as f:
        json.dump(ontology_extractions, f, indent=2)
    LOG.info(f"✅ Saved updated ontology to {updated_ontology_file_path}")
    return updated_ontology_file_path