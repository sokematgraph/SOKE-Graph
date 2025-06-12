# ai_tool/gemini_tool.py
from sokegraph.ai_agent import AIAgent

class GeminiAgent(AIAgent):
    def __init__(self, api_key):
        self.api_key = api_key  # Save key if needed

    def ask(self, prompt: str) -> str:
        # Replace with real Gemini API call
        return f"Gemini response to: {prompt}"
