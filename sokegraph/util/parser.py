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