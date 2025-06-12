import re

from sokegraph import config

def get_next_api_key(api_keys):
    config.counter_index_API = config.counter_index_API + 1
    key = api_keys[(config.counter_index_API + 1) % len(api_keys)]
    return key

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

def parse_all_metadata(ontology_extractions: dict) -> None:
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
    # Normalize formatting
    meta_str = meta_str.replace("cm\u22122", "cm^-2")
    meta_str = meta_str.replace("−", "-").replace("µ", "u").replace("μ", "u")

    # Pattern for numbers + scientific units
    pattern = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Zμµ°²^/%\-]+)"
    return [{"value": float(val), "unit": unit} for val, unit in re.findall(pattern, meta_str)]


def load_keyword(keyword_query_file_path):
    with open(keyword_query_file_path, "r") as file:
        user_query = file.read()
    return user_query