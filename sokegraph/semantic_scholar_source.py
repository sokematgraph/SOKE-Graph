# semantic_scholar_source.py
from semanticscholar import SemanticScholar
from time import sleep
from sokegraph.util.logger import LOG
from typing import List, Dict
from sokegraph.base_paper_source import BasePaperSource

class SemanticScholarPaperSource(BasePaperSource):
    def __init__(self, num_papers: int, query_file: str, output_dir: str):
        self.num_papers = num_papers
        self.query_file = query_file
        self.output_dir = output_dir
        self.sch = SemanticScholar()

    def fetch_papers(self) -> List[Dict]:
        queries = self._load_queries()
        return self._get_unique_papers_from_scholar(queries)

    def _load_queries(self) -> List[str]:
        with open(self.query_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _get_unique_papers_from_scholar(self, queries: List[str]) -> List[Dict]:
        papers = []
        for query in queries:
            if len(papers) >= self.num_papers:
                break
            LOG.info(f"Searching Semantic Scholar for: {query}")
            try:
                results = self.sch.search_paper(query, limit=100)
                for paper in results:
                    if len(papers) >= self.num_papers:
                        break
                    papers.append({
                        "paper_id": paper.paperId,
                        "title": paper.title,
                        "abstract": paper.abstract or "",
                        "authors": ", ".join([a["name"] for a in paper.authors]) if paper.authors else "",
                        "year": paper.year,
                        "venue": paper.venue,
                        "url": paper.url,
                        "doi": paper.externalIds.get("DOI") if paper.externalIds else ""
                    })
                sleep(1)
            except Exception as e:
                LOG.error(f"Error searching '{query}': {e}")
        unique_papers = {p["paper_id"]: p for p in papers}
        papers = list(unique_papers.values())
        self.export_metadata_to_excel(papers, self.output_dir)
        return papers
