from openai import OpenAI
import re

class RelationEvaluatingAgent:
    def __init__(self, api_key, base_url, prompts,model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompts = prompts
        self.model = model

    def extract_score(self, response_message):
        score_pattern = re.compile(r"Score:\s*([0-5])", re.IGNORECASE)
        match = score_pattern.search(response_message)
        if match:
            return int(match.group(1))
        return None

    def evaluate(self, key_phrases, prompt_type):
        prompt = self.prompts[prompt_type]
        max_attempts = 3
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
