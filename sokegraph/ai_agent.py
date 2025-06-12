from abc import ABC, abstractmethod
from typing import List, Dict, Any

class AIAgent(ABC):
    def ask(self, prompt: str) -> str:
        raise NotImplementedError("This method should be overridden.")
    
    @abstractmethod
    def extract_keywords(self, ontology: dict, text_data: dict) -> dict:
        pass

    @abstractmethod
    def get_opposites(self, keywords: List[str]) -> Dict[str, List[str]]:
        pass

    @abstractmethod
    def get_synonyms(self, keywords: List[str]) -> Dict[str, List[str]]:
        pass

    def load_api_keys(self, API_file_path: str):
        # Open the file and read all non-empty lines, stripping whitespace
        with open(API_file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
