import requests
from time import sleep
from typing import List, Dict
from sokegraph.sources.base_paper_source import BasePaperSource
from sokegraph.util.logger import LOG


class JournalApiPaperSource(BasePaperSource):
    """
    Paper source class that fetches journals from the Clarivate Web of Science Journal API
    based on queries loaded from a text file.
    
    Inherits from:
    - BasePaperSource: provides common interface and metadata export utility.
    """

    def __init__(self, num_papers: int, query_file: str, output_dir: str, api_key_file: str):
        """
        Initializes the source with parameters for querying the Journal API.

        Parameters:
        - num_papers (int): Maximum number of journal entries to fetch.
        - query_file (str): Path to a text file containing search queries, one per line.
        - output_dir (str): Directory path to save outputs such as Excel metadata.
        - api_key (str): API key for authenticating with the Web of Science Journal API.
        """
        if not api_key_file:
            raise ValueError("API key is required to access the Web of Science Journal API.")

        self.num_papers = num_papers
        self.query_file = query_file
        self.output_dir = output_dir
        self.api_key = self.load_api_key(api_key_file)
        self.base_url = "https://api.clarivate.com/apis/wos-journals/v1/journals"

    @staticmethod
    def load_api_key(api_key_file):
        
        with open(api_key_file, "r") as file:
            api_key = file.read()
        return api_key
    
    def fetch_papers(self) -> List[Dict]:
        """
        Main method to fetch journals based on queries.

        Returns:
        - List[Dict]: List of journal metadata dictionaries.
        """
        queries = self._load_queries()
        journals = self._get_unique_journals_from_wos(queries)
        self.export_metadata_to_excel(journals, self.output_dir)
        return journals

    def _load_queries(self) -> List[str]:
        """
        Loads search queries from the specified query file.

        Returns:
        - List[str]: List of non-empty queries read from the file.
        """
        with open(self.query_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _get_unique_journals_from_wos(self, queries: List[str]) -> List[Dict]:
        """
        Executes search queries against the Web of Science Journal API and collects unique results.

        Parameters:
        - queries (List[str]): List of search queries.

        Returns:
        - List[Dict]: List of unique journal metadata dictionaries.
        """
        headers = {
            "X-ApiKey": self.api_key,
            "Accept": "application/json"
        }

        collected = {}

        for query in queries:
            if len(collected) >= self.num_papers:
                break

            LOG.info(f"Searching Web of Science Journal API for: {query}")

            try:
                response = requests.get(
                    self.base_url,
                    headers=headers,
                    params={"title": query, "limit": 100}
                )
                response.raise_for_status()
                results = response.json().get("journals", [])

                for item in results:
                    if len(collected) >= self.num_papers:
                        break

                    issn = item.get("issn") or item.get("eissn")
                    if not issn or issn in collected:
                        continue

                    collected[issn] = {
                        "paper_id": issn,
                        "title": item.get("title", ""),
                        "abstract": item.get("aimsAndScope", ""),
                        "authors": "",  # Not applicable at journal level
                        "year": item.get("coverage", {}).get("from", ""),
                        "venue": item.get("publisher", "Unknown Publisher"),
                        "url": item.get("url", ""),
                        "doi": ""  # DOI not provided at journal level
                    }

                sleep(1)  # Respect API rate limits

            except requests.HTTPError as http_err:
                LOG.error(f"HTTP error for '{query}': {http_err.response.status_code} - {http_err.response.text}")
            except Exception as e:
                LOG.exception(f"Unexpected error querying '{query}': {e}")

        return list(collected.values())