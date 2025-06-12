import json
import re
import logging
from collections import defaultdict
from openai import OpenAI
from sokegraph.ai_agent import AIAgent  # abstract class
from sokegraph.functions import get_next_api_key  # assumes load-balancing across keys
from typing import List, Dict, Any

LOG = logging.getLogger(__name__)


class OpenAIAgent(AIAgent):
    def __init__(self, api_keys_path: str):
        
        self.api_keys = self.load_api_keys(api_keys_path)

    def ask(self, prompt: str) -> str:
        """
        Send a general prompt to OpenAI and return the raw response.
        Used internally by _call_openai_model and available for flexible querying.
        """
        try:
            client = OpenAI(api_key=get_next_api_key(self.api_keys))
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            LOG.error(f"❌ OpenAI API call failed in ask(): {e}")
            return ""

    def extract_keywords(self, ontology: dict, text_data: dict) -> dict:
        """
        Extract keywords from paper abstracts using the ontology layers and OpenAI.
        Returns:
            dict[layer][category] = list of {keywords, meta_data, paper_id}
        """
        results_dict = defaultdict(lambda: defaultdict(list))

        for paper_id, abstract in text_data.items():
            LOG.info(f"🔍 Processing paper: {paper_id}")
            for layer, categories in ontology.items():
                results = self._call_openai_model(layer, abstract, categories)

                for res in results:
                    cat = res["matched_category"]
                    if cat not in ontology[layer]:
                        LOG.warning(f"⚠️ Skipping unknown category '{cat}' under layer '{layer}'")
                        continue

                    keywords = list(set(ontology[layer][cat] + [res["keyword"]]))
                    results_dict[layer][cat].append({
                        "keywords": keywords,
                        "meta_data": res["meta_data"],
                        "paper_id": paper_id
                    })

                LOG.info(f"✅ Extracted {len(results)} items for paper {paper_id}, layer '{layer}'")

        return results_dict

    def _call_openai_model(self, layer: str, abstract_text: str, ontology_layer: dict) -> list:
        """
        Internal method that builds prompt, calls ask(), and parses output.
        """
        prompt = self._build_prompt(layer, abstract_text, ontology_layer)
        model_output = self.ask(prompt)
        if not model_output:
            return []
        return self._parse_model_output(model_output)

    def _build_prompt(self, layer_name: str, abstract_text: str, ontology_layer: dict) -> str:
        """
        Build prompt string for the given layer and abstract.
        """
        return f"""
                You are a structured information extraction AI specialized in materials science.

                You will extract keywords relevant to the layer: **{layer_name}**
                For this layer, the following categories and sample keywords are provided:
                {json.dumps(ontology_layer, indent=2)}

                From the text below, extract:
                - Any new or related keywords
                - The category it fits under
                - Any nearby numeric values with units (e.g., overpotential, pH, current density)

                Text to analyze:
                {abstract_text}

                Return the output in this exact JSON list format:
                [
                {{
                    "layer": "LayerName",
                    "matched_category": "Category",
                    "keyword": "ExtractedKeyword",
                    "meta_data": "Text snippet"
                }}
                ]
                """

    def _parse_model_output(self, model_output: str) -> list:
        """
        Try parsing the LLM response as JSON, fallback to regex if needed.
        """
        try:
            parsed = json.loads(model_output)
            LOG.info(f"✅ JSON parsed successfully. Extracted {len(parsed)} items.")
            return parsed
        except json.JSONDecodeError:
            LOG.warning("⚠️ JSON parsing failed. Attempting fallback regex...")

            pattern = r'"layer":\s*"([^"]+)",\s*"matched_category":\s*"([^"]+)",\s*"keyword":\s*"([^"]+)",\s*"meta_data":\s*"([^"]+)"'
            matches = re.findall(pattern, model_output, re.DOTALL)
            extracted = [
                {
                    "layer": m[0].strip(),
                    "matched_category": m[1].strip(),
                    "keyword": m[2].strip(),
                    "meta_data": m[3].strip()
                }
                for m in matches
            ]
            LOG.info(f"✅ Extracted {len(extracted)} items via regex.")
            return extracted
    

    def get_opposites(self, keywords: List[str]) -> Dict[str, List[str]]:
        prompt = f"""
            You are a materials science assistant.

            For each of the following scientific keywords, return a list of their opposite or contradictory concepts 
            in the context of electrocatalysis, water splitting, or electrochemical environments.

            Keywords: {keywords}

            Respond only in JSON format like this:
            {{
              "acidic": ["alkaline", "basic"],
              "HER": ["OER", "oxygen evolution"]
            }}
        """
        return self._parse_json(self._ask(prompt), "opposites")

    def get_synonyms(self, keywords: List[str]) -> Dict[str, List[str]]:
        prompt = f"""
            You are a materials science assistant.

            For each of the following scientific keywords, return a list of their synonyms or semantically 
            similar expressions used in electrocatalysis, water splitting, or electrochemistry papers.

            Keywords: {keywords}

            Respond only in JSON format like this:
            {{
              "acidic": ["low pH", "acid media"],
              "HER": ["hydrogen evolution", "H2 generation"]
            }}
        """
        return self._parse_json(self._ask(prompt), "synonyms")

