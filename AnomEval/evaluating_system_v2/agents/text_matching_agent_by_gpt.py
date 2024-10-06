import sys
import os
import re

# Add the parent directory of 'agents' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
from agents.strategy import MatchingStrategy
from config import OPENAI_BASE_URL, PROMPT_FOR_TEXT_MATCHING, MODEL_TYPE

class TextMatchingAgentByGPT(MatchingStrategy):
    def __init__(self, api_key, base_url, prompt,model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = prompt
        self.model = model

    def extract_score(self, response_message):
        # score_pattern = re.compile(r"Score:\s*([0-5])", re.IGNORECASE)
        score_pattern = re.compile(r"Score:\s*\[?([0-5])\]?", re.IGNORECASE)

        match = score_pattern.search(response_message)
        if match:
            return int(match.group(1))
        return None
    
    def get_text_matching_score(self, features, ground_truth):
        json_data = f'''
          "feature": "{features}",
          "truth": "{ground_truth}"
        '''
        max_attempts = 1000
        attempt = 0

        while attempt < max_attempts:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": json_data}
                    ]
                )
                response_message = completion.choices[0].message.content
                extracted_score = self.extract_score(response_message)
                if extracted_score is not None:
                    #print('attempt:',attempt)
                    return int(extracted_score)
                else:
                    attempt += 1
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                attempt += 1
        print(f"Failed to generate a valid response after {max_attempts} attempts.")
        return None

if __name__ == "__main__":
    OPENAI_API_KEY = 'sk-an0o8YvIYzTmmh7SC5EaE9389bFe42Ac914a2c212e164c33'
    agent = TextMatchingAgentByGPT(OPENAI_API_KEY, OPENAI_BASE_URL, PROMPT_FOR_TEXT_MATCHING, MODEL_TYPE)
    features = "The video"
    ground_truth = "The video shows"
    score = agent.get_text_matching_score(features, ground_truth)
    print(f"Score: {score}")