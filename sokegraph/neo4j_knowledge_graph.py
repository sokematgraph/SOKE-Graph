from py2neo import Graph, Node, Relationship
from collections import defaultdict
from sokegraph.knowledge_graph import KnowledgeGraph

class Neo4jKnowledgeGraph(KnowledgeGraph):
    def __init__(self, ontology_path, uri, user, password):
        super().__init__(ontology_path)
        self.graph = Graph(uri, auth=(user, password))

    def build_graph(self):
        unique_keywords = defaultdict(set)
        unique_categories = set()
        unique_layers = set()
        paper_keywords = defaultdict(set)

        for layer, categories in self.ontology_extractions.items():
            layer_node = self._get_or_create_layer(layer, unique_layers)

            for cat, items in categories.items():
                cat_key = f"{layer}|{cat}"
                cat_node = self._get_or_create_category(cat, cat_key, layer_node, unique_categories)

                for item in items:
                    paper_id = item.get("paper_id", "unknown")
                    for kw in item["keywords"]:
                        kw_key = f"{layer}|{cat}|{kw}"
                        if kw_key in unique_keywords[(layer, cat)]:
                            continue
                        unique_keywords[(layer, cat)].add(kw_key)

                        kw_node = Node("Keyword", name=kw, key=kw_key)
                        self.graph.merge(kw_node, "Keyword", "key")
                        self.graph.merge(Relationship(cat_node, "HAS_KEYWORD", kw_node))

                        for meta in item.get("parsed_meta", []):
                            meta_node = Node("MetaData", value=meta["value"], unit=meta["unit"], type=cat)
                            self.graph.create(meta_node)
                            self.graph.create(Relationship(kw_node, "HAS_METADATA", meta_node))

                        paper_keywords[paper_id].add(kw_key)

        print("✅ Layer, Category, Keyword, and MetaData nodes created")
        self._create_paper_nodes(paper_keywords)
        print("🎉 Knowledge graph construction complete.")

    def _get_or_create_layer(self, layer, unique_layers):
        if layer not in unique_layers:
            node = Node("Layer", name=layer)
            self.graph.merge(node, "Layer", "name")
            unique_layers.add(layer)
        else:
            node = self.graph.nodes.match("Layer", name=layer).first()
        return node

    def _get_or_create_category(self, cat, cat_key, layer_node, unique_categories):
        if cat_key not in unique_categories:
            node = Node("Category", name=cat, key=cat_key)
            self.graph.merge(node, "Category", "key")
            self.graph.merge(Relationship(layer_node, "HAS_CATEGORY", node))
            unique_categories.add(cat_key)
        else:
            node = self.graph.nodes.match("Category", key=cat_key).first()
        return node

    def _create_paper_nodes(self, paper_keywords):
        for paper_id, kw_keys in paper_keywords.items():
            paper_node = Node("Paper", id=paper_id)
            self.graph.merge(paper_node, "Paper", "id")

            for kw_key in kw_keys:
                kw_node = self.graph.nodes.match("Keyword", key=kw_key).first()
                if kw_node:
                    rel = Relationship(kw_node, "MENTIONS", paper_node)
                    self.graph.merge(rel)

        print("✅ Paper nodes and MENTIONS relationships created")
