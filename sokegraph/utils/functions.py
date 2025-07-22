import re
from sokegraph.ontology import config


def get_next_api_key(api_keys):
    """
    Cycle through a list of API keys using a global counter.

    This helps avoid rate-limiting by rotating across multiple keys.

    Args:
        api_keys (list[str]): A list of API key strings.

    Returns:
        str: The next API key in a round-robin fashion.
    """
    config.counter_index_API = config.counter_index_API + 1
    key = api_keys[(config.counter_index_API + 1) % len(api_keys)]
    return key


def safe_title(title: str, max_len: int = 100) -> str:
    """
    Sanitize a string to create a safe filename or title by removing unwanted characters.

    Args:
        title (str): The original title string.
        max_len (int, optional): Maximum allowed length of the result. Default is 100.

    Returns:
        str: A cleaned version of the title containing only alphanumerics, spaces,
             underscores, or hyphens, and trimmed to the specified length.
    """
    # Remove any character not in the allowed set
    cleaned_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title)

    # Truncate to the specified length and remove leading/trailing spaces
    return cleaned_title[:max_len].strip()


def parse_all_metadata(ontology_extractions: dict) -> None:
    """
    Enrich extracted ontology items with parsed numerical metadata such as overpotential and Tafel slope.

    Args:
        ontology_extractions (dict): Nested dict of extractions organized by layer and category.
                                     Each item must have a 'meta_data' field.
    """
    for layer, cats in ontology_extractions.items():
        for cat, items in cats.items():
            for item in items:
                # Parse numerical values and units from metadata string
                item["parsed_meta"] = parse_meta_data(item["meta_data"])
                meta_lower = item["meta_data"].lower()

                # Assign known scientific metrics when detected
                for parsed in item["parsed_meta"]:
                    if "overpotential" in meta_lower:
                        item["overpotential"] = parsed
                    if "tafel" in meta_lower:
                        item["tafel_slope"] = parsed


def parse_meta_data(meta_str: str) -> list[dict]:
    """
    Extract numerical values and scientific units from a text string.

    Args:
        meta_str (str): The raw string containing metadata (e.g., "85 mV", "20 mA/cm²").

    Returns:
        list[dict]: A list of dicts with 'value' (float) and 'unit' (str).
    """
    # Normalize known symbol variants for consistent parsing
    meta_str = meta_str.replace("cm\u22122", "cm^-2")
    meta_str = meta_str.replace("−", "-").replace("µ", "u").replace("μ", "u")

    # Regex pattern to match numbers and their units (with scientific notation support)
    pattern = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Zμµ°²^/%\-]+)"
    return [{"value": float(val), "unit": unit} for val, unit in re.findall(pattern, meta_str)]


def load_keyword(keyword_query_file_path):
    """
    Load a keyword query from a plain text file.

    Args:
        keyword_query_file_path (str): Path to the file containing the keyword query.

    Returns:
        str: The entire contents of the file as a single string.
    """
    with open(keyword_query_file_path, "r") as file:
        user_query = file.read()
    return user_query
