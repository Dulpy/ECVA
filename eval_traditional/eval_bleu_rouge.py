import nltk
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')

class RougeScore:
    precision : float = 0
    recall : float = 0
    fmeasure : float = 0

def get_bleu_score(reference, candidate):
    references = [[word for word in reference.split()]]
    candidate_segmented = [word for word in candidate.split()]
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu(references, candidate_segmented, smoothing_function=smoothie)
    return bleu_score

def get_rouge_score(reference, candidate):
    references = [[word for word in reference.split()]]
    candidate_segmented = [word for word in candidate.split()]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def main():
    description_file = '/result/chat_description.json'
    reason_file = 'result/chat_reason.json'
    outcome_file = 'result/chat_outcome.json'
    
    index = 0
    with open(description_file, 'r') as fp:
        data = json.load(fp)
        for item in data:
            if item['task_type'] == 'default':
                index += 1
    
    index = 985 # TODO FIX

    des_bleu = 0
    des_rouge = {
        'rouge1': RougeScore(),
        'rouge2': RougeScore(),
        'rougeL': RougeScore()
    }
    with open(description_file, 'r') as fp:
        f = json.load(fp)
        for data in f[:index]:
            response = data['output']
            answer = data['human_expert_answer']
            des_bleu += get_bleu_score(response, answer)
            rouge_res = get_rouge_score(response, answer)
            des_rouge['rouge1'].precision += rouge_res['rouge1'].precision
            des_rouge['rouge1'].recall += rouge_res['rouge1'].recall
            des_rouge['rouge1'].fmeasure += rouge_res['rouge1'].fmeasure
            des_rouge['rouge2'].precision += rouge_res['rouge2'].precision
            des_rouge['rouge2'].recall += rouge_res['rouge2'].recall
            des_rouge['rouge2'].fmeasure += rouge_res['rouge2'].fmeasure
            des_rouge['rougeL'].precision += rouge_res['rougeL'].precision
            des_rouge['rougeL'].recall += rouge_res['rougeL'].recall
            des_rouge['rougeL'].fmeasure += rouge_res['rougeL'].fmeasure
    print('description bleu:', des_bleu / index)
    print('description rouge1:',des_rouge['rouge1'].fmeasure / index)
    print('description rouge2:',des_rouge['rouge2'].fmeasure / index)
    print('description rougeL:',des_rouge['rougeL'].fmeasure / index)

    reason_bleu = 0
    reason_rouge = {
        'rouge1': RougeScore(),
        'rouge2': RougeScore(),
        'rougeL': RougeScore()
    }
    with open(reason_file, 'r') as fp:
        f = json.load(fp)
        for data in f[:index]:
            response = data['output']
            answer = data['human_expert_answer']
            reason_bleu += get_bleu_score(response, answer)
            rouge_res = get_rouge_score(response, answer)
            reason_rouge['rouge1'].precision += rouge_res['rouge1'].precision
            reason_rouge['rouge1'].recall += rouge_res['rouge1'].recall
            reason_rouge['rouge1'].fmeasure += rouge_res['rouge1'].fmeasure
            reason_rouge['rouge2'].precision += rouge_res['rouge2'].precision
            reason_rouge['rouge2'].recall += rouge_res['rouge2'].recall
            reason_rouge['rouge2'].fmeasure += rouge_res['rouge2'].fmeasure
            reason_rouge['rougeL'].precision += rouge_res['rougeL'].precision
            reason_rouge['rougeL'].recall += rouge_res['rougeL'].recall
            reason_rouge['rougeL'].fmeasure += rouge_res['rougeL'].fmeasure
    print('reason bleu:', reason_bleu / index)
    print('reason rouge1:',reason_rouge['rouge1'].fmeasure / index)
    print('reason rouge2:',reason_rouge['rouge2'].fmeasure / index)
    print('reason rougeL:',reason_rouge['rougeL'].fmeasure / index)

    outcome_bleu = 0
    outcome_rouge = {
        'rouge1': RougeScore(),
        'rouge2': RougeScore(),
        'rougeL': RougeScore()
    }
    with open(outcome_file, 'r') as fp:
        f = json.load(fp)
        for data in f[:index]:
            response = data['output']
            answer = data['human_expert_answer']
            outcome_bleu += get_bleu_score(response, answer)
            rouge_res = get_rouge_score(response, answer)
            outcome_rouge['rouge1'].precision += rouge_res['rouge1'].precision
            outcome_rouge['rouge1'].recall += rouge_res['rouge1'].recall
            outcome_rouge['rouge1'].fmeasure += rouge_res['rouge1'].fmeasure
            outcome_rouge['rouge2'].precision += rouge_res['rouge2'].precision
            outcome_rouge['rouge2'].recall += rouge_res['rouge2'].recall
            outcome_rouge['rouge2'].fmeasure += rouge_res['rouge2'].fmeasure
            outcome_rouge['rougeL'].precision += rouge_res['rougeL'].precision
            outcome_rouge['rougeL'].recall += rouge_res['rougeL'].recall
            outcome_rouge['rougeL'].fmeasure += rouge_res['rougeL'].fmeasure
    print('outcome bleu:', outcome_bleu / index)
    print('outcome rouge1:',outcome_rouge['rouge1'].fmeasure / index)
    print('outcome rouge2:',outcome_rouge['rouge2'].fmeasure / index)
    print('outcome rougeL:',outcome_rouge['rougeL'].fmeasure / index)



if __name__ == '__main__':
    main()