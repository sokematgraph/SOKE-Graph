# ai_tool/gemini_tool.py
from sokegraph.ai_agent import AIAgent
from itertools import cycle
from together import Together
from sokegraph.functions import get_next_api_key


class LlamaAgent(AIAgent):
    def __init__(self, api_keys_path: str):
        self.api_keys = self.load_api_keys(api_keys_path)
        #self.key_cycle = cycle(self.api_keys)
    

    def ask(self, prompt: str, max_retries=3) -> str:
        for attempt in range(max_retries):
            try:
                #key = next(self.key_cycle)
                client = Together(api_key=get_next_api_key(self.api_keys))

                response = client.chat.completions.create(
                    #model="meta-llama/Llama-3-70b-instruct",  # ← Confirm model name here
                    #model = "meta-llama/Llama-2-70b-chat-hf",
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"⚠️ LLaMA error (attempt {attempt + 1}): {e}")
                continue

        raise RuntimeError("❌ All LLaMA API attempts failed.")