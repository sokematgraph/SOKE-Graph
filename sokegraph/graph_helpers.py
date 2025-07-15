from neo4j import GraphDatabase

def get_driver(creds: dict):
    return GraphDatabase.driver(
        creds["uri"],
        auth=(creds["user"], creds["password"])
    )

def get_neighbor_subgraph(tx, node_id: int, max_hops: int = 1, limit: int = 50):
    query = f"""
    MATCH (n)-[r*1..{max_hops}]-(m)
    WHERE id(n) = $id
    WITH apoc.coll.toSet(n + m) AS nodes,
         r AS rels
    RETURN nodes, rels
    LIMIT $limit
    """
    rec = tx.run(query, id=node_id, limit=limit).single()
    return rec["nodes"], rec["rels"]


def fetch_related_nodes_and_edges(driver, node_name: str) -> list[dict]:
    """
    For a node with a given name, fetch directly connected neighbor nodes,
    their relationship types, and direction (outgoing/incoming).
    """
    query = """
    MATCH (n {name: $name})-[r]-(m)
    RETURN DISTINCT
           m,
           type(r) AS rel_type,
           startNode(r) = n AS outgoing
    """

    with driver.session() as session:
        results = session.run(query, name=node_name)
        return [
            {
                "node": record["m"],
                "rel_type": record["rel_type"],
                "direction": "out" if record["outgoing"] else "in"
            }
            for record in results
        ]
