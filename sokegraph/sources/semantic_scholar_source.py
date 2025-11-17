from semanticscholar import SemanticScholar
from time import sleep
from sokegraph.util.logger import LOG
from typing import List, Dict
from sokegraph.sources.base_paper_source import BasePaperSource
import pandas as pd
from sokegraph.sources.abstract_helper import abstract_from_paper
from sokegraph.sources.venue_helper import venue_from_title

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import requests
from tenacity import RetryError as TenacityRetryError


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

    def _get_unique_papers_from_scholar(self, queries: List[str]) -> str:
        """
        Executes search queries against Semantic Scholar and collects unique papers.
        """
        papers = []

        @retry(
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type((requests.exceptions.ConnectionError, TenacityRetryError, ConnectionRefusedError))
        )
        def safe_search(query):
            """Try searching Semantic Scholar safely with retries."""
            LOG.info(f"Searching Semantic Scholar for: {query}")
            return self.sch.search_paper(
                query,
                limit=100,
                fields=[
                    "paperId", "title", "abstract", "authors", "year", "venue", "url",
                    "externalIds", "citationStyles", "citationCount"
                ]
            )

        for query in queries:
            if len(papers) >= self.num_papers:
                break

            try:
                results = safe_search(query)
                for p in results:
                    if len(papers) >= self.num_papers:
                        break

                    authors = ", ".join([a["name"] for a in (p.authors or [])]) if hasattr(p, "authors") else ""
                    doi = ""
                    if hasattr(p, "externalIds") and isinstance(p.externalIds, dict):
                        doi = p.externalIds.get("DOI", "")

                    bibtex = ""
                    if hasattr(p, "citationStyles") and isinstance(p.citationStyles, dict):
                        bibtex = p.citationStyles.get("bibtex", "") or ""

                    citation_count = getattr(p, "citationCount", None)

                    papers.append({
                        "paper_id": getattr(p, "paperId", ""),
                        "title": getattr(p, "title", "") or "",
                        "authors": authors,
                        "year": getattr(p, "year", None),
                        "venue": getattr(p, "venue", "") or "",
                        "abstract": (
                            getattr(p, "abstract", None)
                            or getattr(p, "_abstract", None)
                            or getattr(p, "Abstract", None)
                            or ""
                        ),
                        "url": getattr(p, "url", "") or "",
                        "doi": doi,
                        "bibtex": bibtex,
                        "citation_count": citation_count,
                        "source": "Semantic Scholar",
                    })
                    print(f"DEBUG: Retrieved paper '{getattr(p, 'title', '')}'")

                sleep(1)  # avoid rate-limit
            except Exception as e:
                LOG.error(f"Error searching '{query}': {e}")
                sleep(10)  # wait before next query

        # Deduplicate
        unique = {p["paper_id"]: p for p in papers if p.get("paper_id")}
        papers = list(unique.values())

        pd.DataFrame(papers).to_excel("debug_semantic_scholar_papers.xlsx", index=False)

        for paper in papers:
            if pd.isna(paper["abstract"]) or not str(paper["abstract"]).strip():
                res = abstract_from_paper(paper=paper)
                paper["abstract"] = res["abstract"]
                paper["source"] = res["source"]

            if pd.isna(paper["venue"]) or not str(paper["venue"]).strip():
                title = paper["title"]
                res = venue_from_title(title)
                paper["venue"] = res["venue"]
                paper["year"] = res["year"]
                paper["doi"] = res["doi"]

        return self.export_metadata_to_excel(papers, self.output_dir)
