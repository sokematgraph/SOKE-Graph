"""
semantic_scholar_source.py

Implements paper retrieval from the Semantic Scholar API.

This module provides :class:`SemanticScholarPaperSource`, which:
- Loads search queries from a text file
- Fetches papers from Semantic Scholar API with deduplication
- Enriches missing abstracts and venue information
- Exports results to Excel format

Key Features:
- Real-time deduplication during collection
- Rate limiting (3-second delay between queries) to avoid HTTP 429 errors  
- Automatic abstract/venue enrichment for missing data
- Continues collecting across multiple queries until target count reached

Example Usage:
    >>> source = SemanticScholarPaperSource(
    ...     num_papers=40,
    ...     query_file="queries.txt",
    ...     output_dir="./output"
    ... )
    >>> excel_path = source.fetch_papers()
"""
from semanticscholar import SemanticScholar
from time import sleep
from sokegraph.util.logger import LOG
from typing import List, Dict
from sokegraph.sources.base_paper_source import BasePaperSource
import pandas as pd
from sokegraph.sources.abstract_helper import abstract_from_title
from sokegraph.sources.venue_helper import venue_from_title

# Rate limiting configuration to avoid API throttling
SEMANTIC_SCHOLAR_REQUEST_DELAY = 3  # seconds between API requests

class SemanticScholarPaperSource(BasePaperSource):
    """Fetches research papers from Semantic Scholar API.
    
    This class handles paper retrieval with automatic deduplication,
    rate limiting, and data enrichment for missing fields.
    
    Attributes
    ----------
    num_papers : int
        Target number of unique papers to retrieve
    query_file : str
        Path to text file containing search queries (one per line)
    output_dir : str
        Directory where output Excel file will be saved
    sch : SemanticScholar
        Semantic Scholar API client instance
    """
    
    def __init__(self, num_papers: int, query_file: str, output_dir: str):
        """Initialize the Semantic Scholar paper source.
        
        Parameters
        ----------
        num_papers : int
            Maximum number of unique papers to retrieve
        query_file : str
            Path to text file containing search queries (one per line)
        output_dir : str
            Directory where output Excel file will be saved
        """
        self.num_papers = num_papers
        self.query_file = query_file
        self.output_dir = output_dir
        self.sch = SemanticScholar()

    def fetch_papers(self) -> List[Dict]:
        """Fetch papers from Semantic Scholar API.
        
        Loads queries from the query file and retrieves unique papers
        across all queries until the target count is reached.
        
        Returns
        -------
        str
            Path to the Excel file containing fetched paper metadata
        """
        queries = self._load_queries()
        return self._get_unique_papers_from_scholar(queries)

    def _load_queries(self) -> List[str]:
        """Load search queries from the query file.
        
        Returns
        -------
        List[str]
            List of non-empty search query strings
            
        Raises
        ------
        FileNotFoundError
            If query_file does not exist
        """
        with open(self.query_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _get_unique_papers_from_scholar(self, queries: List[str]) -> str:
        """
        Executes search queries against Semantic Scholar and collects unique papers.

        Parameters:
        - queries (List[str]): List of search queries.

        Returns:
        - List[Dict]: List of unique paper metadata dictionaries.
        """
        papers = []
        for query in queries:
            if len(papers) >= self.num_papers:
                break

            LOG.info(f"Searching Semantic Scholar for: {query}")
            try:
                results = self.sch.search_paper(
                    query, limit=100,
                    fields=[
                        "paperId", "title", "abstract", "authors", "year", "venue", "url",
                        "externalIds", "citationStyles", "citationCount"
                    ]
                )
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

                    citation_count = None
                    if hasattr(p, "citationCount"):
                        citation_count = getattr(p, "citationCount", None)

                    papers.append({
                        "paper_id": getattr(p, "paperId", ""),
                        "title": getattr(p, "title", "") or "",
                        "authors": authors,
                        "year": getattr(p, "year", None),
                        "venue": getattr(p, "venue", "") or "",
                        "abstract": getattr(p, "abstract", "") or "",
                        "url": getattr(p, "url", "") or "",
                        "doi": doi,
                        "bibtex": bibtex,
                        "citation_count": citation_count,
                    })
                sleep(SEMANTIC_SCHOLAR_REQUEST_DELAY)
            except Exception as e:
                LOG.error(f"Error searching '{query}': {e}")

        unique = {p["paper_id"]: p for p in papers if p.get("paper_id")}
        papers = list(unique.values())
        for paper in papers:
            if pd.isna(paper["abstract"]) or not str(paper["abstract"]).strip():
                title = paper["title"]
                res = abstract_from_title(title, use_openai=False)
                paper["abstract"] = res['abstract']
            
            if pd.isna(paper["venue"]) or not str(paper["venue"]).strip():
                title = paper["title"]
                res = venue_from_title(title)
                paper["venue"] = res['venue']
                paper["year"] = res['year']
                paper["doi"] = res['doi']
        

        return self.export_metadata_to_excel(papers, self.output_dir)
