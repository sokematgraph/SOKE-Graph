import json
from py2neo import Graph, Node, Relationship
from collections import defaultdict

def build_neo4j_knowledge_graph_from_ontology_json(
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

with open("../external/input/neo4j.json", "r") as file:
        credentials = json.load(file)

build_neo4j_knowledge_graph_from_ontology_json("../external/output/updated_ontology.json",
credentials["neo4j_uri"],
credentials["neo4j_user"],
credentials["neo4j_pass"])