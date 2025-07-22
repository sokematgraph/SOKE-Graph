"""
deepseek_agent.py
Concrete implementation of :class:`sokegraph.ai_agent.AIAgent`
that queries DeepSeek-R1. Includes simple API-key rotation.
"""
from __future__ import annotations
import logging
from typing import List
from openai import OpenAI  # This SDK wraps the OpenAI-compatible DeepSeek endpoint
from sokegraph.agents.ai_agent import AIAgent
from sokegraph.utils.functions import get_next_api_key
LOG = logging.getLogger(__name__)
class DeepSeekAgent(AIAgent):
    """
    Wrapper around DeepSeek-R1 via the OpenAI-compatible ChatCompletion API.
    Parameters
    ----------
    api_keys_path : str
        Path to a plaintext file containing one DeepSeek API key per line.
    """
    def __init__(self, api_keys_path: str):
        self.api_keys: List[str] = self.load_api_keys(api_keys_path)
    def load_api_keys(self, path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    def ask(self, prompt: str) -> str:
        """
        Send a single prompt to the DeepSeek-R1 model.
        Parameters
        ----------
        prompt : str
            The full text prompt to send to the model.
        Returns
        -------
        str
            The model’s textual response, or an empty string on failure.
        """
        try:
            api_key = get_next_api_key(self.api_keys)
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"  # :white_check_mark: Updated base URL per docs
            )
            response = client.chat.completions.create(
                model="deepseek-chat",  # :white_check_mark: This is the correct model for DeepSeek-R1
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            LOG.error(":x: DeepSeek-R1 API call failed in ask(): %s", e)
            return ""