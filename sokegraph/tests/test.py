from sokegraph.paper_ranker import PaperRanker
from sokegraph.base_paper_source import BasePaperSource
from sokegraph.semantic_scholar_source import SemanticScholarPaperSource
from sokegraph.openai_agent import OpenAIAgent

ai_tool = OpenAIAgent("external/input/APIs.txt")
# 1. Select paper source
paper_source: BasePaperSource
paper_source = SemanticScholarPaperSource(
    num_papers=int(200),
    query_file="external/input/paper_query.txt",
    output_dir="external/output"
)
    

papers = paper_source.fetch_papers()
ranker = PaperRanker(ai_tool, papers, "external/output/updated_ontology.json", "external/input/keyword_query.txt", "external/output")
updated_ontology_path = ranker.rank_papers()