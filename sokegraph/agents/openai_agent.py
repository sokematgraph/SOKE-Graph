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
from openai._exceptions import OpenAIError, APIStatusError, RateLimitError
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

    def __init__(self, api_keys_path, field_of_interest):
        super().__init__(field_of_interest=field_of_interest)
        # Load and store all available keys for rotation
        self.api_keys: List[str] = self.load_api_keys(api_keys_path)
        # Rotate to the next key to spread usage across accounts
        self.client = OpenAI(api_key=get_next_api_key(self.api_keys))
        
        

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    
    def ask(self, prompt: str) -> str:
        """
        Send a single prompt to the GPT-4o model.

        Implements a very lightweight key-rotation scheme via
        :func:`sokegraph.functions.get_next_api_key`.

        Parameters
        ----------
        prompt : str
            The full text prompt to send to the model.

        Returns
        -------
        str
            The model’s textual response, or an empty list on failure.
        """
        # Rotate to the next key to spread usage across accounts
        client = OpenAI(api_key=get_next_api_key(self.api_keys))
        print(f"Using API key: {client.api_key}")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a structured information extraction AI specialized in materials science."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            model_output = response.choices[0].message.content.strip()
            print("📄 OpenAI returned:\n", model_output[:500])  # Show preview only
            
            return model_output

        except RateLimitError:
            LOG.error("⚠️ Rate limit exceeded. Please try again later.")
            return ""
        except APIStatusError as e:
            LOG.error("❌ API returned an error ({e.status_code}): {e.message}")
            return ""
        except OpenAIError as e:
            LOG.error("❌ An unexpected OpenAI error occurred: {str(e)}")
            return ""
        except Exception as e:
            LOG.error("❌ An unexpected system error occurred: {str(e)}")
            return ""
