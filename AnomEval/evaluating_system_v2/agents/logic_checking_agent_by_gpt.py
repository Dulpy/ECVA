import sys
import os

# Add the parent directory of 'agents' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI
from config import OPENAI_BASE_URL,MODEL_TYPE,PROMPTS_FOR_REASONING_ABILITY_CHECKING
import re

class LogicCheckingAgent:
    def __init__(self, api_key, base_url, prompt,model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = prompt
        self.model = model

    def extract_score(self, response_message):
        score_pattern = re.compile(r"Score:\s*([0-5])", re.IGNORECASE)
        match = score_pattern.search(response_message)
        if match:
            return int(match.group(1))
        return None

    def get_logic_score(self, key_phrases):
        prompt = self.prompt
        max_attempts = 1000
        attempt = 0

        while attempt < max_attempts:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": key_phrases}
                    ]
                )
                response_message = completion.choices[0].message.content
                score = self.extract_score(response_message)
                if score is not None:
                    return score
                else:
                    attempt += 1
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                attempt += 1
        
        print(f"Failed to generate a valid response after {max_attempts} attempts.")
        return None
    


