# ai_tool/gemini_tool.py

# Import base AI agent interface
from sokegraph.agents.ai_agent import AIAgent

# Used for cycling through API keys (currently not in use)
from itertools import cycle

# Together API client for interacting with language models
from together import Together

# Helper function to rotate or retrieve the next API key from a list
from sokegraph.utils.functions import get_next_api_key


class LlamaAgent(AIAgent):
    """
    A concrete implementation of the AIAgent class using the Together API
    to query the LLaMA language model.
    """
    def __init__(self,  api_keys_path, field_of_interest):
        super().__init__(field_of_interest=field_of_interest)
        # Load API keys from the given path using the parent method
        self.api_keys = self.load_api_keys(api_keys_path)

        # Optional: Create a cycling iterator for keys if needed later
        # self.key_cycle = cycle(self.api_keys)

    def ask(self, prompt: str, max_retries=3) -> str:
        """
        Send a prompt to the LLaMA model and return the generated response.
        
        Args:
            prompt (str): The user prompt to send to the model.
            max_retries (int): Number of times to retry in case of failure.

        Returns:
            str: The model's response as a string.

        Raises:
            RuntimeError: If all attempts fail to get a response.
        """
        for attempt in range(max_retries):
            try:
                # Optionally use round-robin key selection
                # key = next(self.key_cycle)

                # Get the next available API key using a custom helper
                client = Together(api_key=get_next_api_key(self.api_keys))

                # Send request to Together's LLaMA model
                response = client.chat.completions.create(
                    # Uncomment and adjust model name if needed
                    # model="meta-llama/Llama-3-70b-instruct",
                    # model = "meta-llama/Llama-2-70b-chat-hf",
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Free version of LLaMA 3.3 model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,  # Lower temperature = more deterministic output
                    max_tokens=50   # Limit response length
                )

                # Extract and return the model's response text
                return response.choices[0].message.content.strip()

            except Exception as e:
                # Log any errors and retry
                print(f"⚠️ LLaMA error (attempt {attempt + 1}): {e}")
                continue

        # If all retries fail, raise an exception
        raise RuntimeError("❌ All LLaMA API attempts failed.")