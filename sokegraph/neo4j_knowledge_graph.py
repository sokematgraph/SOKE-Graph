from py2neo import Graph, Node, Relationship
from collections import defaultdict
from sokegraph.knowledge_graph import KnowledgeGraph

class Neo4jKnowledgeGraph(KnowledgeGraph):
    """
    Concrete implementation of KnowledgeGraph that builds a graph in Neo4j.

    Nodes created:
        - Layer
        - Category
        - Keyword
        - MetaData
        - Paper

    Relationships created:
        - Layer HAS_CATEGORY Category
        - Category HAS_KEYWORD Keyword
        - Keyword HAS_METADATA MetaData
        - Keyword MENTIONS Paper
    """

    def __init__(self, ontology_path: str, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection and load ontology extractions.

        Parameters
        ----------
        ontology_path : str
            Path to ontology JSON file.
        uri : str
            Neo4j database URI (e.g., bolt://localhost:7687).
        user : str
            Neo4j username.
        password : str
            Neo4j password.
        """
        super().__init__(ontology_path)
        self.graph = Graph(uri, auth=(user, password))

    def build_graph(self):
        """
        Build the knowledge graph in Neo4j from the loaded ontology extractions.

        Avoids duplications by tracking created nodes with sets.
        Creates nodes and relationships in the following order:
        Layers → Categories → Keywords → MetaData → Papers
        """
        unique_keywords = defaultdict(set)  # Track keywords by (layer, category)
        unique_categories = set()            # Track categories by cat_key
        unique_layers = set()                # Track created layer names
        paper_keywords = defaultdict(set)   # Map paper_id to related keyword keys

        # Iterate through ontology: layer → categories → extracted items
        for layer, categories in self.ontology_extractions.items():
            layer_node = self._get_or_create_layer(layer, unique_layers)

            for cat, items in categories.items():
                cat_key = f"{layer}|{cat}"
                cat_node = self._get_or_create_category(cat, cat_key, layer_node, unique_categories)

                for item in items:
                    paper_id = item.get("paper_id", "unknown")

                    # Process keywords, avoiding duplicates within layer/category
                    for kw in item["keywords"]:
                        kw_key = f"{layer}|{cat}|{kw}"
                        if kw_key in unique_keywords[(layer, cat)]:
                            continue
                        unique_keywords[(layer, cat)].add(kw_key)

                        # Create or merge Keyword node and relationship to Category
                        kw_node = Node("Keyword", name=kw, key=kw_key)
                        self.graph.merge(kw_node, "Keyword", "key")
                        self.graph.merge(Relationship(cat_node, "HAS_KEYWORD", kw_node))

                        # Create MetaData nodes linked to Keyword
                        for meta in item.get("parsed_meta", []):
                            meta_node = Node(
                                "MetaData",
                                value=meta["value"],
                                unit=meta["unit"],
                                type=cat
                            )
                            self.graph.create(meta_node)
                            self.graph.create(Relationship(kw_node, "HAS_METADATA", meta_node))

                        # Associate paper with this keyword
                        paper_keywords[paper_id].add(kw_key)

        print("✅ Layer, Category, Keyword, and MetaData nodes created")
        self._create_paper_nodes(paper_keywords)
        print("🎉 Knowledge graph construction complete.")

    def _get_or_create_layer(self, layer: str, unique_layers: set) -> Node:
        """
        Retrieve existing or create new Layer node in Neo4j.

        Parameters
        ----------
        layer : str
            Layer name.
        unique_layers : set
            Set tracking which layers have been created.

        Returns
        -------
        Node
            The Layer node.
        """
        if layer not in unique_layers:
            node = Node("Layer", name=layer)
            self.graph.merge(node, "Layer", "name")
            unique_layers.add(layer)
        else:
            node = self.graph.nodes.match("Layer", name=layer).first()
        return node

    def _get_or_create_category(self, cat: str, cat_key: str, layer_node: Node, unique_categories: set) -> Node:
        """
        Retrieve existing or create new Category node linked to the given Layer node.

        Parameters
        ----------
        cat : str
            Category name.
        cat_key : str
            Unique key combining layer and category.
        layer_node : Node
            Parent Layer node.
        unique_categories : set
            Set tracking created categories.

        Returns
        -------
        Node
            The Category node.
        """
        if cat_key not in unique_categories:
            node = Node("Category", name=cat, key=cat_key)
            self.graph.merge(node, "Category", "key")
            self.graph.merge(Relationship(layer_node, "HAS_CATEGORY", node))
            unique_categories.add(cat_key)
        else:
            node = self.graph.nodes.match("Category", key=cat_key).first()
        return node

    def _create_paper_nodes(self, paper_keywords: dict):
        """
        Create Paper nodes and MENTIONS relationships to Keyword nodes.

        Parameters
        ----------
        paper_keywords : dict
            Mapping from paper ID to set of keyword keys mentioned in that paper.
        """
        for paper_id, kw_keys in paper_keywords.items():
            paper_node = Node("Paper", id=paper_id)
            self.graph.merge(paper_node, "Paper", "id")

            for kw_key in kw_keys:
                kw_node = self.graph.nodes.match("Keyword", key=kw_key).first()
                if kw_node:
                    rel = Relationship(kw_node, "MENTIONS", paper_node)
                    self.graph.merge(rel)

        print("✅ Paper nodes and MENTIONS relationships created")