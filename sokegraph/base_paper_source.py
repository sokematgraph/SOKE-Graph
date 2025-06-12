# base_paper_source.py
from abc import ABC, abstractmethod
from typing import List, Dict
import os
import pandas as pd

class BasePaperSource(ABC):
    @abstractmethod
    def fetch_papers(self) -> List[Dict]:
        """Retrieve papers with metadata"""
        pass

    def export_metadata_to_excel(self, papers: List[Dict], output_dir: str, filename: str = "papers_metadata.xlsx"):
        if not papers:
            return
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(papers)
        filepath = os.path.join(output_dir, filename)
        df.to_excel(filepath, index=False)
