from sokegraph.base_paper_source import BasePaperSource
from sokegraph.semantic_scholar_source import SemanticScholarPaperSource
from sokegraph.pdf_paper_source import PDFPaperSource
from sokegraph.paper_ranker import PaperRanker
from sokegraph.knowledge_graph import KnowledgeGraph
from sokegraph.util.logger import LOG
from sokegraph.ai_agent import AIAgent
from sokegraph.openai_agent import OpenAIAgent
from sokegraph.gemini_agent import GeminiAgent
from sokegraph.ontology_updater import OntologyUpdater
from sokegraph.neo4j_knowledge_graph import Neo4jKnowledgeGraph
from sokegraph.llama_agent import LlamaAgent

import json

def full_pipeline_main(params):
    LOG.info("🚀 Starting Full Pipeline")

    # 0. Setup AI agent
    ai_tool: AIAgent
    if params.AI == "openAI":
        ai_tool = OpenAIAgent(params.API_keys)
    elif params.AI == "gemini":
        ai_tool = GeminiAgent(params.API_keys)
    elif params.AI == "llama":
        ai_tool = LlamaAgent(params.API_keys)
    else:
        raise ValueError(f"Unsupported AI provider: {params.AI}")

    # 1. Select paper source
    paper_source: BasePaperSource
    if params.number_papers and not params.pdfs_file:
        if not params.paper_query_file:
            LOG.error("❌ paper_query_file is required when using number_papers.")
            return
        paper_source = SemanticScholarPaperSource(
            num_papers=int(params.number_papers),
            query_file=params.paper_query_file,
            output_dir=params.output_dir
        )
    elif params.pdfs_file and not params.number_papers:
        paper_source = PDFPaperSource(
            zip_path=params.pdfs_file,
            output_dir=params.output_dir
        )
    else:
        LOG.error("❌ Please specify either number_papers or pdfs_file, but not both.")
        return

    papers = paper_source.fetch_papers()

    # 2. Update ontology
    #ontology_updater = OntologyUpdater(params.ontology_file, papers, ai_tool, params.output_dir)  # or however you instantiate it
    #ontology_extractions = ontology_updater.enrich_with_papers()


    #LOG.info("ranking papers ....")
    # 3. Rank papers
    ranker = PaperRanker(ai_tool, papers, f"{params.output_dir}/updated_ontology.json", params.keyword_query_file, params.output_dir)
    updated_ontology_path = ranker.rank_papers()

    # 4. Build knowledge graph
    #LOG.info(" Building knowledge graph ....")
    #with open(params.credentials_for_knowledge_graph, "r") as f:
    #    credentials = json.load(f)

    #graph_builder: KnowledgeGraph
    #if(params.model_knowledge_graph == "neo4j"):
    #    graph_builder = Neo4jKnowledgeGraph(f"{params.output_dir}/updated_ontology.json", 
    #                                        credentials["neo4j_uri"],
    #                                        credentials["neo4j_user"],
    #                                        credentials["neo4j_pass"])
    
    #graph_builder.build_graph()

    LOG.info("🎉 Pipeline Completed Successfully")
