import json
import os

from bleurt import score

checkpoint = "../eval_mertic/BLEURT-20"

scorer = score.BleurtScorer(checkpoint)


def main():
    description_file = 'result/chat_description.json'
    reason_file = 'result/chat_reason.json'
    outcome_file = 'result/chat_outcome.json'

    count_index = 985 # TODO FIX
    default_num = 0
    with open(description_file, 'r') as f:
        data = json.load(f)
        for item in data:
            if item['task_type'] == 'default':
                default_num += 1
    if default_num:
        count_index = 1000 - default_num
    des_score = 0

    with open(description_file, 'r') as f:
        data = json.load(f)
        for item in data[:count_index]:
            references = [item['output']]
            candidates = [item['human_expert_answer']]
            scores = scorer.score(references=references, candidates=candidates)
            print(scores)
            des_score += scores[0]

    print("des_score",des_score / count_index)

    cause_score = 0
    with open(reason_file, 'r') as f:
        data = json.load(f)
        for item in data[:count_index]:
            references = [item['output']]
            candidates = [item['human_expert_answer']]
            scores = scorer.score(references=references, candidates=candidates)
            print(scores)
            cause_score += scores[0]

    print("cause_score",cause_score / count_index)



    outcome_score = 0
    with open(outcome_file, 'r') as f:
        data = json.load(f)
        for item in data[:count_index]:
            references = [item['output']]
            candidates = [item['human_expert_answer']]
            scores = scorer.score(references=references, candidates=candidates)
            print(scores)
            outcome_score += scores[0]

    print("outcome_score",outcome_score / count_index)

    print(description_file)

    print("des_score",des_score / count_index)
    print("cause_score",cause_score / count_index)
    print("outcome_score",outcome_score / count_index)

if __name__ == '__main__':
    main()