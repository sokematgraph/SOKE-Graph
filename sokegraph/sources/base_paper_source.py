"""
base_paper_source.py

Defines an abstract base class for all paper retrieval sources.

Subclasses must implement `fetch_papers()` to return paper metadata
from a given source (e.g., Semantic Scholar, PDF files, etc.).

Also provides a shared utility to export metadata to Excel.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import os
import pandas as pd


class BasePaperSource(ABC):
    """Abstract base class for retrieving paper metadata.

    All paper source classes should inherit from this class and
    implement the `fetch_papers` method.
    """

    @abstractmethod
    def fetch_papers(self) -> List[Dict]:
        """Fetch and return a list of papers with metadata.

        Each paper should be a dictionary with standardized keys
        such as: 'title', 'authors', 'year', 'abstract', etc.

        Returns:
            List of dictionaries, one per paper.
        """
        pass

    def export_metadata_to_excel(self,
                                 papers: List[Dict],
                                 output_dir: str,
                                 filename: str = "papers_metadata.xlsx") -> str:
        """Export paper metadata to an Excel (.xlsx) file.

        Args:
            papers: List of paper metadata dictionaries.
            output_dir: Folder where the Excel file will be saved.
            filename: Name of the output Excel file (default is 'papers_metadata.xlsx').

        Notes:
            If the papers list is empty, the function does nothing.
        """
        if not papers:
            return  "" # Nothing to export

        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        df = pd.DataFrame(papers)              # Convert list of dicts to DataFrame
        filepath = os.path.join(output_dir, filename)
        df.to_excel(filepath, index=False)     # Export to Excel
        return filepath
