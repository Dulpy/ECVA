import sys
import os

# Add the parent directory of 'agents' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI
from config import OPENAI_BASE_URL, PROMPTS_FOR_REASONING_ABILITY_CHECKING
import re

class ReasoningAbilityCheckingAgent:
    def __init__(self, api_key, base_url, prompt,model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = prompt
        self.model = model

    def extract_score(self, response_message):
        score_pattern = re.compile(r"Score:\s*([0-5])", re.IGNORECASE)
        match = score_pattern.search(response_message)
        if match:
            return int(match.group(1))
        return None

    def get_reasoning_ability_score(self, key_phrases):
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
                # print(response_message)
                score = self.extract_score(response_message)
                # print(score)
                if score is not None:
                    #print('attempt:',attempt)
                    return score
                else:
                    attempt += 1
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                attempt += 1
        
        print(f"Failed to generate a valid response after {max_attempts} attempts.")
        return None
    



if __name__ == "__main__":
    OPENAI_API_KEY = 'sk-an0o8YvIYzTmmh7SC5EaE9389bFe42Ac914a2c212e164c33'
    logic_checking_agent = ReasoningAbilityCheckingAgent(OPENAI_API_KEY, OPENAI_BASE_URL, PROMPTS_FOR_REASONING_ABILITY_CHECKING)
    print('PROMPT:'+PROMPTS_FOR_REASONING_ABILITY_CHECKING)
    key_phrases_0 = 'None'
    key_phrases_1 = "none"
    key_phrases_2 = "None"
    key_phrases_3 = "none"
    key_phrases_4 = "None!"
    for i in range(1):
        answer_0 = logic_checking_agent.get_reasoning_ability_score(key_phrases_0)
        answer_1 = logic_checking_agent.get_reasoning_ability_score(key_phrases_1)
        answer_2 = logic_checking_agent.get_reasoning_ability_score(key_phrases_2)
        answer_3 = logic_checking_agent.get_reasoning_ability_score(key_phrases_3)
        answer_4 = logic_checking_agent.get_reasoning_ability_score(key_phrases_4)
        print(f"Answer 0: {answer_0}")
        print(f"Answer 1: {answer_1}")
        print(f"Answer 2: {answer_2}")
        print(f"Answer 3: {answer_3}")
        print(f"Answer 4: {answer_4}")