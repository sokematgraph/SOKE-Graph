import json
from sokegraph.util.logger import LOG
from sokegraph.paper_ranking_openAI import ranking_papers
from sokegraph.input_handling import *
from sokegraph.update_ontology_openAI import enrich_ontology_with_papers_openAI, load_ontology
from sokegraph.build_neo4j_knowledge_graph import build_neo4j_knowledge_graph_from_ontology_json

import re


counter_index_API = 0

def full_pipeline_main(params):
    LOG.info("Starirng Ranking ...")
    
    papers = get_papers(params=params)
    
    text_data, title_map, abstract_map = load_paper_data(papers=papers)
    
    API_keys = load_api_keys(params.API_keys)
    
    if params.AI=="openAI":
        updated_ontology_file_path =  enrich_ontology_with_papers_openAI(params.ontology_file, API_keys, text_data, params.output_dir)
    
        ontology_extractions = load_ontology(updated_ontology_file_path)

        with open(params.keyword_query_file, "r") as file:
            keyword_query = file.read()

        ranking_papers(API_keys,
            ontology_extractions,
            title_map,
            abstract_map,
            keyword_query,
            params.output_dir)

    
    with open(params.credentials_for_knowledge_graph, "r") as file:
        credentials = json.load(file)

    if params.model_knowledge_graph=="neo4j":
        build_neo4j_knowledge_graph_from_ontology_json(updated_ontology_file_path,
        credentials["neo4j_uri"],
        credentials["neo4j_user"],
        credentials["neo4j_pass"])