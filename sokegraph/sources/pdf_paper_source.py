# pdf_paper_source.py

import zipfile, os, requests
from PyPDF2 import PdfReader
from sokegraph.util.logger import LOG
from sokegraph.utils.utils import check_file
from typing import List, Dict
from sokegraph.sources.base_paper_source import BasePaperSource

class PDFPaperSource(BasePaperSource):
    """
    A class to handle PDF-based paper retrieval by:
    1. Unzipping PDFs.
    2. Extracting titles from metadata.
    3. Querying the Semantic Scholar API using the extracted titles.
    4. Returning structured paper metadata.

    Inherits from:
    - BasePaperSource: common interface for paper sources.
    """

    def __init__(self, zip_path: str, output_dir: str):
        """
        Initialize with a path to a zip file of PDFs and an output directory.

        Parameters:
        - zip_path (str): Path to the zip file containing PDFs.
        - output_dir (str): Directory where extracted PDFs and output will be stored.
        """
        self.zip_path = check_file(zip_path)  # Validates the zip file exists
        self.output_dir = output_dir

    def fetch_papers(self) -> List[Dict]:
        """
        Main method to extract and fetch paper metadata.

        Returns:
        - List[Dict]: List of paper metadata dictionaries retrieved from Semantic Scholar.
        """
        pdf_paths = self._unzip_pdfs()  # Step 1: Unzip PDFs
        papers = []

        # Step 2: Process each PDF
        for pdf in pdf_paths:
            LOG.info(f"Processing {pdf}")
            title = self._extract_title_from_pdf(pdf)  # Step 3: Extract title from metadata
            if title:
                info = self._query_semantic_scholar(title)  # Step 4: Search Semantic Scholar
                if info:
                    papers.append(info)

        # Step 5: Save paper metadata to Excel (inherited from BasePaperSource)
        self.export_metadata_to_excel(papers, self.output_dir)
        return papers

    def _unzip_pdfs(self) -> List[str]:
        """
        Unzips the PDFs from the provided zip file.

        Returns:
        - List[str]: Paths to the extracted PDF files.
        """
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            extract_path = f"{self.output_dir}/extracted_pdfs"
            zip_ref.extractall(extract_path)

        # Collect all PDF files from the extracted folder
        return [os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.endswith(".pdf")]

    def _extract_title_from_pdf(self, pdf_path: str) -> str:
        """
        Attempts to extract the document title from PDF metadata.

        Parameters:
        - pdf_path (str): Path to the PDF file.

        Returns:
        - str: Extracted title or empty string if unavailable or failed.
        """
        try:
            reader = PdfReader(pdf_path)
            metadata = reader.metadata or {}
            return metadata.get("/Title", "Unknown Title")
        except Exception as e:
            LOG.error(f"Error reading PDF metadata from {pdf_path}: {e}")
            return ""

    def _query_semantic_scholar(self, title: str) -> Dict:
        """
        Uses Semantic Scholar's API to retrieve metadata based on the paper title.

        Parameters:
        - title (str): Paper title to search for.

        Returns:
        - Dict: Dictionary with metadata like abstract, authors, year, DOI, etc.
                Returns None if the API request fails or yields no result.
        """
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search?"
            f"query={title}&limit=1&fields=title,abstract,authors,year,venue,url,externalIds"
        )
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data["data"]:
                    paper = data["data"][0]
                    return {
                        "paper_id": paper.get("paperId"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract", ""),
                        "authors": ", ".join([a["name"] for a in paper.get("authors", [])]),
                        "year": paper.get("year"),
                        "venue": paper.get("venue"),
                        "url": paper.get("url"),
                        "doi": paper.get("externalIds", {}).get("DOI", "")
                    }
        except Exception as e:
            LOG.error(f"Error querying Semantic Scholar for title '{title}': {e}")
        return None