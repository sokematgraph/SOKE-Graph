# ai_tool/gemini_tool.py 

from sokegraph.agents.ai_agent import AIAgent
import google.generativeai as genai
from sokegraph.utils.functions import get_next_api_key

class GeminiAgent(AIAgent):
    """
    GeminiAgent is a subclass of AIAgent that interfaces with Google's Gemini LLM
    for structured information extraction tasks in materials science.
    """

    def __init__(self, api_keys_path, field_of_interest):
        super().__init__(field_of_interest=field_of_interest)
        """
        Initialize the GeminiAgent by loading API keys from a file.

        Args:
            api_keys_path (str): Path to the file containing API keys (one per line).
        """
        self.api_keys = self.load_api_keys(api_keys_path)

    def ask(self, prompt: str) -> str:
        """
        Send a prompt to the Gemini model and return the response text.

        This method handles API key rotation using `get_next_api_key()` and
        calls Google's Gemini 2.0 Flash model.

        Args:
            prompt (str): The prompt string to be sent to the LLM.

        Returns:
            str: The textual response from the model. If the call fails, returns an empty string.
        """
        # Configure the API client with the next key to balance usage
        genai.configure(api_key=get_next_api_key(self.api_keys))

        # Initialize the Gemini model (flash version is faster and cheaper)
        model = genai.GenerativeModel("gemini-2.0-flash")

        try:
            # Generate a response to the input prompt
            response = model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            # Graceful fallback on API failure
            print(f"❌ Gemini API call failed: {e}")
            return ""
