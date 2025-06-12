# pdf_paper_source.py
import zipfile, os, requests
from PyPDF2 import PdfReader
from sokegraph.util.logger import LOG
from sokegraph.utils import check_file
from typing import List, Dict
from sokegraph.base_paper_source import BasePaperSource

class PDFPaperSource(BasePaperSource):
    def __init__(self, zip_path: str, output_dir: str):
        self.zip_path = check_file(zip_path)
        self.output_dir = output_dir

    def fetch_papers(self) -> List[Dict]:
        pdf_paths = self._unzip_pdfs()
        papers = []
        for pdf in pdf_paths:
            LOG.info(f"Processing {pdf}")
            title = self._extract_title_from_pdf(pdf)
            if title:
                info = self._query_semantic_scholar(title)
                if info:
                    papers.append(info)
        self.export_metadata_to_excel(papers, self.output_dir)
        return papers

    def _unzip_pdfs(self) -> List[str]:
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            extract_path = f"{self.output_dir}/extracted_pdfs"
            zip_ref.extractall(extract_path)
        return [os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.endswith(".pdf")]

    def _extract_title_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            metadata = reader.metadata or {}
            return metadata.get("/Title", "Unknown Title")
        except Exception as e:
            LOG.error(f"Error reading PDF metadata from {pdf_path}: {e}")
            return ""

    def _query_semantic_scholar(self, title: str) -> Dict:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields=title,abstract,authors,year,venue,url,externalIds"
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
