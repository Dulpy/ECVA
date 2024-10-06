# main.py
import os
from config import OPENAI_API_KEY, OPENAI_BASE_URL, PROMPTS_FOR_RELATION_EVALUATING

def main():
    print("OPENAI_API_KEY:", OPENAI_API_KEY)
    print("OPENAI_BASE_URL:", OPENAI_BASE_URL)
    print("PROMPTS_FOR_RELATION_EVALUATING:", PROMPTS_FOR_RELATION_EVALUATING)

if __name__ == "__main__":
    main()