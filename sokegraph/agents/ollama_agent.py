# sokegraph/ollama_agent.py

# Import the requests library to make HTTP requests to the local Ollama server
import requests

# Import the base class AIAgent that this class will extend
from sokegraph.agents.ai_agent import AIAgent

class OllamaAgent(AIAgent):
    """
    AI Agent that interfaces with a locally running Ollama server (e.g., llama3 model).
    This class is useful for offline natural language processing using models like LLaMA.
    """

    def __init__(self, field_of_interest, model="llama3", base_url="http://localhost:11434"):
        
        """
        Initialize the OllamaAgent.

        Parameters:
        - field_of_interest (str): User's field of interest
        - field_of_interest (str): User's field of interest
        - model (str): The name of the model to use (e.g., "llama3")
        - base_url (str): The base URL where the Ollama server is running
        """
        super().__init__(field_of_interest=field_of_interest)
        self.model = model
        self.base_url = base_url

    def ask(self, prompt: str) -> str:
        """
        Sends a prompt to the Ollama server and returns the generated response.

        Parameters:
        - prompt (str): The text prompt to send to the local LLM

        Returns:
        - str: The generated response from the Ollama model or an error message
        """
        try:
            # Make a POST request to Ollama's /generate endpoint with model and prompt
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False  # non-streaming mode (get full response at once)
                },
                timeout=60  # optional timeout for the request
            )

            # Parse the JSON response and return the text content
            result = response.json()
            return result.get("response", "No response received.")

        except Exception as e:
            # Handle connection errors or unexpected issues gracefully
            return f"❌ Error communicating with Ollama: {str(e)}"
