"""
openai_agent.py

Concrete implementation of :class:`sokegraph.ai_agent.AIAgent`
that queries OpenAI’s GPT‑4o model.  Includes simple API‑key
rotation to mitigate rate‑limit issues.
"""

from __future__ import annotations

import json
import re
import logging
from typing import List, Dict, Any

from openai import OpenAI
from sokegraph.agents.ai_agent import AIAgent
from sokegraph.utils.functions import get_next_api_key

LOG = logging.getLogger(__name__)


class OpenAIAgent(AIAgent):
    """
    Wrapper around OpenAI’s ChatCompletion API that satisfies the AIAgent interface.

    Parameters
    ----------
    api_keys_path : str
        Path to a plaintext file containing one OpenAI key per line.
    """

    def __init__(self, api_keys_path: str):
        # Load and store all available keys for rotation
        self.api_keys: List[str] = self.load_api_keys(api_keys_path)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def ask(self, prompt: str) -> str:
        """
        Send a single prompt to the GPT‑4o model.

        Implements a very lightweight key‑rotation scheme via
        :func:`sokegraph.functions.get_next_api_key`.

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
            # Rotate to the next key to spread usage across accounts
            client = OpenAI(api_key=get_next_api_key(self.api_keys))

            # Call the chat completion endpoint
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.0  # deterministic output for extraction
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            LOG.error("❌ OpenAI API call failed in ask(): %s", e)
            return ""
