from abc import ABC, abstractmethod

class MatchingStrategy(ABC):
    @abstractmethod
    def get_text_matching_score(self, features, ground_truth):
        pass

class GPTMatchingStrategy(MatchingStrategy):
    def __init__(self, client, prompt):
        self.client = client
        self.prompt = prompt

    def get_text_matching_score(self, features, ground_truth,model="gpt-4o-mini"):
        json_data = f'''{{
          "feature": "{features}",
          "truth": "{ground_truth}"
        }}'''
        
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": json_data}
            ]
        )
        response_message = completion.choices[0].message.content
        return response_message

class BERTMatchingStrategy(MatchingStrategy):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def get_text_matching_score(self, features, ground_truth):
        features_embedding = self.get_bert_embeddings(features)
        ground_truth_embedding = self.get_bert_embeddings(ground_truth)
        
        similarity_score = cosine_similarity(features_embedding, ground_truth_embedding)[0][0]
        return similarity_score
