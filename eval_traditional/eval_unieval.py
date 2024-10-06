import json
import sys
import os
os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
sys.path.append('../eval_mertic/UniEval')

from utils import convert_to_json
from metric.evaluator import get_evaluator

task = 'fact'
# Initialize evaluator for a specific task
evaluator = get_evaluator(task)
# Get factual consistency scores

data = convert_to_json(output_list=output_list, src_list=src_list)

eval_scores = evaluator.evaluate(data, print_result=True)

def main():
    des_file = 'result/chat_description.json'
    reason_file = 'result/chat_reason.json'
    outcome_file = 'result/chat_outcome.json'

    index = 985 # TODO FIX
    default_num = 0
    with open(des_file, 'r') as f:
        data = json.load(f)
        for item in data:
            if item['task_type'] == 'default':
                default_num += 1
    if default_num:
        index = 1000 - default_num

    des_score, reason_score, outcome_score = 0, 0, 0
    with open(des_file, 'r') as f:
        data = json.load(f)
        for item in data[:index]:
            src_list = [item['human_expert_answer']]
            output_list = [item['output']]
            data = convert_to_json(output_list=output_list, src_list=src_list)
            eval_scores = evaluator.evaluate(data, print_result=True)
            des_score += eval_scores[0]['consistency']

    with open(reason_file, 'r') as f:
        data = json.load(f)
        for item in data[:index]:
            src_list = [item['human_expert_answer']]
            output_list = [item['output']]
            data = convert_to_json(output_list=output_list, src_list=src_list)
            eval_scores = evaluator.evaluate(data, print_result=True)
            reason_score += eval_scores[0]['consistency']

    with open(outcome_file, 'r') as f:
        data = json.load(f)
        for item in data[:index]:
            src_list = [item['human_expert_answer']]
            output_list = [item['output']]
            data = convert_to_json(output_list=output_list, src_list=src_list)
            eval_scores = evaluator.evaluate(data, print_result=True)
            outcome_score += eval_scores[0]['consistency']

    print('des_score:', des_score / index)
    print('reason_score:', reason_score / index)
    print('outcome_score:', outcome_score / index)

if __name__ == '__main__':
    main()