"""
claude_agent.py
Concrete implementation of :class:`sokegraph.ai_agent.AIAgent`
that queries Anthropic’s Claude model. Includes simple API-key
rotation to mitigate rate-limit issues.
"""
from __future__ import annotations
import logging
from typing import List
from anthropic import Anthropic
from sokegraph.agents.ai_agent import AIAgent
from sokegraph.utils.functions import get_next_api_key
LOG = logging.getLogger(__name__)
class ClaudeAgent(AIAgent):
    """
    Wrapper around Anthropic’s Claude API that satisfies the AIAgent interface.
    Parameters
    ----------
    api_keys_path : str
        Path to a plaintext file containing one Anthropic key per line.
    """
    def __init__(self, api_keys_path, field_of_interest):
        super().__init__(field_of_interest=field_of_interest)
        self.api_keys: List[str] = self.load_api_keys(api_keys_path)
    
    def ask(self, prompt: str) -> str:
        """
        Send a single prompt to Claude.
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
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514", # change the model id to: claude-3-haiku-20240307
                max_tokens=1024,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            LOG.error(":x: Claude API call failed in ask(): %s", e)
            return ""