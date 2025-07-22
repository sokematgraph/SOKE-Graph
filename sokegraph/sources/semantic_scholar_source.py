# semantic_scholar_source.py

from semanticscholar import SemanticScholar
from time import sleep
from sokegraph.util.logger import LOG
from typing import List, Dict
from sokegraph.sources.base_paper_source import BasePaperSource

class SemanticScholarPaperSource(BasePaperSource):
    """
    Paper source class that fetches papers from the Semantic Scholar API
    based on queries loaded from a text file.

    Inherits from:
    - BasePaperSource: providing a common interface and utility methods.
    """

    def __init__(self, num_papers: int, query_file: str, output_dir: str):
        """
        Initializes the source with parameters for querying Semantic Scholar.

        Parameters:
        - num_papers (int): Maximum number of papers to fetch.
        - query_file (str): Path to a text file containing search queries, one per line.
        - output_dir (str): Directory path to save outputs such as Excel metadata.
        """
        self.num_papers = num_papers
        self.query_file = query_file
        self.output_dir = output_dir
        self.sch = SemanticScholar()  # Semantic Scholar API client instance

    def fetch_papers(self) -> List[Dict]:
        """
        Main method to fetch papers based on queries.

        Returns:
        - List[Dict]: List of paper metadata dictionaries from Semantic Scholar.
        """
        queries = self._load_queries()  # Load queries from file
        return self._get_unique_papers_from_scholar(queries)  # Fetch and return unique papers

    def _load_queries(self) -> List[str]:
        """
        Loads search queries from the specified query file.

        Returns:
        - List[str]: List of non-empty queries read from the file.
        """
        with open(self.query_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]  # Strip whitespace and ignore empty lines

    def _get_unique_papers_from_scholar(self, queries: List[str]) -> List[Dict]:
        """
        Executes search queries against Semantic Scholar and collects unique papers.

        Parameters:
        - queries (List[str]): List of search queries.

        Returns:
        - List[Dict]: List of unique paper metadata dictionaries.
        """
        papers = []

        # Loop through queries until desired number of papers is reached
        for query in queries:
            if len(papers) >= self.num_papers:
                break

            LOG.info(f"Searching Semantic Scholar for: {query}")

            try:
                # Search papers matching the query, limit to 100 results per query
                results = self.sch.search_paper(query, limit=100)

                for paper in results:
                    if len(papers) >= self.num_papers:
                        break

                    # Collect relevant paper metadata; can be extended with journal, citations, etc.
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

                # Be polite to API: wait 1 second between queries
                sleep(1)

            except Exception as e:
                LOG.error(f"Error searching '{query}': {e}")

        # Remove duplicates by paper_id (keep last occurrence)
        unique_papers = {p["paper_id"]: p for p in papers}
        papers = list(unique_papers.values())

        # Export metadata to Excel (method from BasePaperSource)
        self.export_metadata_to_excel(papers, self.output_dir)

        return papers