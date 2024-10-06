# 从配置文件中导入必要的参数和密钥
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_TYPE, PROMPT_FOR_TEXT_MATCHING, PROMPT_FOR_LOGIC_CHECKING, PROMPT_FOR_KEY_INFOMATION_CHECKING, PROMPTS_FOR_REASONING_ABILITY_CHECKING
from agents.text_matching_agent_by_gpt import TextMatchingAgentByGPT
from agents.logic_checking_agent_by_gpt import LogicCheckingAgent
from agents.key_information_checking_agent import KeyInformationCheckingAgent
from agents.reasoning_ability_checking_agent import ReasoningAbilityCheckingAgent
from utils.data_loader import load_data_from_json_file, load_data_from_xlsx_file
from tqdm import tqdm
import json
import os
import concurrent.futures
import re


def save_progress(progress_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(progress_data, f, indent=4)


def load_progress(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def extract_number_from_string(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return None


def get_fused_data(base_info_list, base_info_name, data_from_VLM_list, data_from_VLM_name):
    fused_data = []
    for base_info, data_from_VLM in zip(base_info_list, data_from_VLM_list):
        if data_from_VLM['output'] != 'ERROR':
            fused_entry = {
                'base infomation': base_info[f'{base_info_name}'],
                'system output': data_from_VLM[f'{data_from_VLM_name}']
            }
            fused_entry_str = str(fused_entry)  
            fused_data.append(fused_entry_str)
        else:
            fused_data.append('ERROR')
    return fused_data

def get_fused_data_with_key_information(base_info_list, base_info_name, data_from_VLM_list, data_from_VLM_name):
    fused_data = []
    for base_info, data_from_VLM in zip(base_info_list, data_from_VLM_list):
        if data_from_VLM['output'] != 'ERROR':
            fused_entry = {
                'key infomation': base_info[f'{base_info_name}'],
                'system output': data_from_VLM[f'{data_from_VLM_name}']
            }
            fused_entry_str = str(fused_entry)  
            fused_data.append(fused_entry_str)
        else:
            fused_data.append('ERROR')
    return fused_data


def get_text_matching_score(agent, key1, key2):
    return int(agent.get_text_matching_score(key1, key2))


def get_logic_score(agent, data):
    return int(agent.get_logic_score(data))


def get_key_information_checking_score(agent, data):
    return agent.get_key_information_checking_score(data)


def get_reasoning_ability_score(agent, data):
    return agent.get_reasoning_ability_score(data)


def main():
    save_dir = '/models/ours/eval_lunwen'
    progress_file = os.path.join(save_dir, 'progress.json')
    progress_data = load_progress(progress_file)
    
    if progress_data is None:
        progress_data = {
            'matching_score': [],
            'logic_score': [],
            'key_information_score': [],
            'reasoning_ability_score': [],
            'current_step': 'matching_score',
            'index_of_matching_score': 0,
            'index_of_logic_score': 0,
            'index_of_key_information_score': 0,
            'index_of_reasoning_ability_score': 0
        }

    
    text_matching_agent = TextMatchingAgentByGPT(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, prompt=PROMPT_FOR_TEXT_MATCHING, model=MODEL_TYPE)
    logic_checking_agent = LogicCheckingAgent(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, prompt=PROMPT_FOR_LOGIC_CHECKING, model=MODEL_TYPE)
    key_information_checking_agent = KeyInformationCheckingAgent(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, prompt=PROMPT_FOR_KEY_INFOMATION_CHECKING, model=MODEL_TYPE)
    reasoning_ability_checking_agent = ReasoningAbilityCheckingAgent(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, prompt=PROMPTS_FOR_REASONING_ABILITY_CHECKING, model=MODEL_TYPE)

    
    reason_data_from_VLM = load_data_from_json_file(r"ours_lunwen_reason.json")
    description_data_from_VLM = load_data_from_json_file(r"ours_lunwen_description.json")
    outcome_data_from_VLM = load_data_from_json_file(r"ours_lunwen_outcome.json")
    data_from_human_expert = load_data_from_xlsx_file(r"ground_truth/annotations.xlsx")
    new_reason_data_from_VLM = load_data_from_json_file(r"chatunivi_new_reason.json")
    new_description_data_from_VLM = load_data_from_json_file(r"chatunivi_new_description.json")
    new_outcome_data_from_VLM = load_data_from_json_file(r"chatunivi_new_outcome.json")

    
    fused_reason_data_list = get_fused_data(base_info_list=data_from_human_expert, base_info_name='X2 - reason（导致异常发生的原因）', data_from_VLM_list=reason_data_from_VLM, data_from_VLM_name='output')
    fused_description_data_list = get_fused_data(base_info_list=data_from_human_expert, base_info_name='X3 - description', data_from_VLM_list=description_data_from_VLM, data_from_VLM_name='output')
    fused_outcome_data_list = get_fused_data(base_info_list=data_from_human_expert, base_info_name='X2 -  result （异常事件导致的结果）', data_from_VLM_list=outcome_data_from_VLM, data_from_VLM_name='output')
    fused_reason_data_for_key_information_checking_list = get_fused_data_with_key_information(base_info_list=data_from_human_expert, base_info_name='X4 - key sentences  （尊重事实的前提下，尽可能写一些（4-6个）独立短句（不超过7个单词），中间不要加逗号）', data_from_VLM_list=reason_data_from_VLM, data_from_VLM_name='output')
    fused_description_data_for_key_information_checking_list = get_fused_data_with_key_information(base_info_list=data_from_human_expert, base_info_name='X4 - key sentences  （尊重事实的前提下，尽可能写一些（4-6个）独立短句（不超过7个单词），中间不要加逗号）', data_from_VLM_list=description_data_from_VLM, data_from_VLM_name='output')
    fused_outcome_data_for_key_information_checking_list = get_fused_data_with_key_information(base_info_list=data_from_human_expert, base_info_name='X4 - key sentences  （尊重事实的前提下，尽可能写一些（4-6个）独立短句（不超过7个单词），中间不要加逗号）', data_from_VLM_list=outcome_data_from_VLM, data_from_VLM_name='output')
    print('data_from_human_expert[0]:', data_from_human_expert[0])
    print("_________________________________________________________")
    print('reason_data_from_VLM[0]:', reason_data_from_VLM[0])
    print("_________________________________________________________")

    print('description_data_from_VLM[0]:', description_data_from_VLM[0])
    print("_________________________________________________________")

    print('outcome_data_from_VLM[0]:', outcome_data_from_VLM[0])
    print("_________________________________________________________")

    print('fused_reason_data_list[0]:', fused_reason_data_list[0])
    print("_________________________________________________________")

    print('fused_description_data_list[0]:', fused_description_data_list[0])
    print("_________________________________________________________")

    print('fused_outcome_data_list[0]:', fused_outcome_data_list[0])
    print("_________________________________________________________")

    print('fused_reason_data_for_key_information_checking_list[0]:', fused_reason_data_for_key_information_checking_list[0])
    print("_________________________________________________________")

    print('fused_description_data_for_key_information_checking_list[0]:', fused_description_data_for_key_information_checking_list[0])
    print("_________________________________________________________")

    print('fused_outcome_data_for_key_information_checking_list[0]:', fused_outcome_data_for_key_information_checking_list[0])
    print("_________________________________________________________")

    print('new_reason_data_from_VLM[0]:', new_reason_data_from_VLM[0])
    print("_________________________________________________________")

    print('new_description_data_from_VLM[0]:', new_description_data_from_VLM[0])
    print("_________________________________________________________")

    print('new_outcome_data_from_VLM[0]:', new_outcome_data_from_VLM[0])
    print('-----------------------------------')

    print('Len of data_from_human_expert:', len(data_from_human_expert))
    print('Len of reason_data_from_VLM:', len(reason_data_from_VLM))
    print('Len of description_data_from_VLM:', len(description_data_from_VLM))
    print('Len of outcome_data_from_VLM:', len(outcome_data_from_VLM))
    print('Len of fused_reason_data_list:', len(fused_reason_data_list))
    print('Len of fused_description_data_list:', len(fused_description_data_list))
    print('Len of fused_outcome_data_list:', len(fused_outcome_data_list))
    print('Len of fused_reason_data_for_key_information_checking_list:', len(fused_reason_data_for_key_information_checking_list))
    print('Len of fused_description_data_for_key_information_checking_list:', len(fused_description_data_for_key_information_checking_list))
    print('Len of fused_outcome_data_for_key_information_checking_list:', len(fused_outcome_data_for_key_information_checking_list))
    print('Len of new_reason_data_from_VLM:', len(new_reason_data_from_VLM))
    print('Len of new_description_data_from_VLM:', len(new_description_data_from_VLM))
    print('Len of new_outcome_data_from_VLM:', len(new_outcome_data_from_VLM))

    print('fused_reason_data_list[938]:', fused_reason_data_list[938])
    print('fused_reason_data_list[939]:', fused_reason_data_list[939])
    print('fused_reason_data_list[940]:', fused_reason_data_list[940])
    
    step_order = ['matching_score', 'logic_score', 'key_information_score', 'reasoning_ability_score']
    current_step = progress_data['current_step']
    start_index_of_matching_score = progress_data['index_of_matching_score']
    start_index_of_logic_score = progress_data['index_of_logic_score']
    start_index_of_key_information_score = progress_data['index_of_key_information_score']
    start_index_of_reasoning_ability_score = progress_data['index_of_reasoning_ability_score']

    res = []
    res_del = []

    for step in step_order[step_order.index(current_step):]:
        if step == 'matching_score':
            print("len(reason_data_from_VLM))", len(reason_data_from_VLM))
            print("start_index_of_matching_score",start_index_of_matching_score)
            for i in tqdm(range(start_index_of_matching_score, len(reason_data_from_VLM))):
                if reason_data_from_VLM[i]['task_type'] == 'default':
                    pass
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # 使用线程池并行计算文本匹配得分
                        future_reason = executor.submit(get_text_matching_score, text_matching_agent, reason_data_from_VLM[i][r'output'], data_from_human_expert[i][r'X2 - reason（导致异常发生的原因）'])
                        future_description = executor.submit(get_text_matching_score, text_matching_agent, description_data_from_VLM[i][r'output'], data_from_human_expert[i][r'X3 - description'])
                        future_outcome = executor.submit(get_text_matching_score, text_matching_agent, outcome_data_from_VLM[i][r'output'], data_from_human_expert[i][r'X2 -  result （异常事件导致的结果）'])

                        # 获取计算结果
                        text_matching_score_for_reason = future_reason.result()
                        text_matching_score_for_description = future_description.result()
                        text_matching_score_for_outcome = future_outcome.result()

                    # 计算平均匹配得分
                    average_matching_score = (text_matching_score_for_reason + text_matching_score_for_description + text_matching_score_for_outcome) / 3
                    sum_matching_score = text_matching_score_for_reason + text_matching_score_for_description + text_matching_score_for_outcome
                    text_matching_score = {
                        'text_matching_score_for_reason': text_matching_score_for_reason,
                        'text_matching_score_for_description': text_matching_score_for_description,
                        'text_matching_score_for_outcome': text_matching_score_for_outcome,
                        'average_matching_score': average_matching_score,
                        'sum_matching_score': sum_matching_score,
                        'note': 'each score is 0 or 1',
                        'video_id': i,
                        'reason_data_from_VLM':reason_data_from_VLM[i][r'output'],
                        'description_data_from_VLM':description_data_from_VLM[i][r'output'],
                        'outcome_data_from_VLM':outcome_data_from_VLM[i][r'output'],
                    }
                    progress_data['matching_score'].append(text_matching_score)

                    # 保存进度
                    progress_data['current_step'] = 'matching_score'
                    progress_data['index_of_matching_score'] = i
                    save_progress(progress_data, progress_file)
            progress_data['current_step'] = 'logic_score'
            progress_data['index_of_logic_score'] = progress_data['index_of_matching_score']
            save_progress(progress_data, progress_file)

        elif step == 'logic_score':
            print("len(fused_reason_data_list))", len(fused_reason_data_list))
            for i in tqdm(range(start_index_of_logic_score, len(fused_reason_data_list))):
                if reason_data_from_VLM[i]['task_type'] == 'default':
                    pass
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_reason = executor.submit(get_logic_score, logic_checking_agent, fused_reason_data_list[i])
                        future_description = executor.submit(get_logic_score, logic_checking_agent, fused_description_data_list[i])
                        future_outcome = executor.submit(get_logic_score, logic_checking_agent, fused_outcome_data_list[i])

                        logic_score_for_reason = future_reason.result()
                        logic_score_for_description = future_description.result()
                        logic_score_for_outcome = future_outcome.result()

                    average_logic_score = (logic_score_for_reason + logic_score_for_description + logic_score_for_outcome) / 3
                    sum_logic_score = logic_score_for_reason + logic_score_for_description + logic_score_for_outcome
                    score = {
                        'logic_score_for_reason': logic_score_for_reason,
                        'logic_score_for_description': logic_score_for_description,
                        'logic_score_for_outcome': logic_score_for_outcome,
                        'average_logic_score': average_logic_score,
                        'sum_logic_score': sum_logic_score,
                        'note': 'each score is from 0 to 5',
                        'video_id': i,
                        'fused_reason_data_list':fused_reason_data_list[i],
                        'fused_description_data_list':fused_description_data_list[i],
                        'fused_outcome_data_list':fused_outcome_data_list[i],
                    }
                    progress_data['logic_score'].append(score)
                    # 保存进度
                    progress_data['current_step'] = 'logic_score'
                    progress_data['index_of_logic_score'] = i
                    save_progress(progress_data, progress_file)
            progress_data['current_step'] = 'key_information_score'
            progress_data['index_of_key_information_score'] = progress_data['index_of_key_information_score']
            save_progress(progress_data, progress_file)

        elif step == 'key_information_score':
            print("len(fused_reason_data_for_key_information_checking_list))", len(fused_reason_data_for_key_information_checking_list))
            for i in tqdm(range(start_index_of_key_information_score, len(fused_reason_data_for_key_information_checking_list))):
                if reason_data_from_VLM[i]['task_type'] == 'default':
                    pass
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_reason = executor.submit(get_key_information_checking_score, key_information_checking_agent, fused_reason_data_for_key_information_checking_list[i])
                        future_description = executor.submit(get_key_information_checking_score, key_information_checking_agent, fused_description_data_for_key_information_checking_list[i])
                        future_outcome = executor.submit(get_key_information_checking_score, key_information_checking_agent, fused_outcome_data_for_key_information_checking_list[i])

                        key_information_score_for_reason = future_reason.result()
                        key_information_score_for_description = future_description.result()
                        key_information_score_for_outcome = future_outcome.result()

                    avg_key_information_score = (key_information_score_for_reason + key_information_score_for_description + key_information_score_for_outcome) / 3
                    sum_key_information_score = key_information_score_for_reason + key_information_score_for_description + key_information_score_for_outcome
                    score = {
                        'key_information_score_for_reason': key_information_score_for_reason,
                        'key_information_score_for_description': key_information_score_for_description,
                        'key_information_score_for_outcome': key_information_score_for_outcome,
                        'average_key_information_score': avg_key_information_score,
                        'sum_key_information_score': sum_key_information_score,
                        'note': 'each score is from 0 to 5',
                        'video_id': i,
                        'fused_reason_data_for_key_information_checking_list':fused_reason_data_for_key_information_checking_list[i],
                        'fused_description_data_for_key_information_checking_list':fused_description_data_for_key_information_checking_list[i],
                        'fused_outcome_data_for_key_information_checking_list':fused_outcome_data_for_key_information_checking_list[i],
                    }
                    progress_data['key_information_score'].append(score)
                    
                    progress_data['current_step'] = 'key_information_score'
                    progress_data['index_of_key_information_score'] = i
                    save_progress(progress_data, progress_file)
            progress_data['current_step'] = 'reasoning_ability_score'
            progress_data['index_of_reasoning_ability_score'] = progress_data['index_of_reasoning_ability_score']
            save_progress(progress_data, progress_file)

        elif step == 'reasoning_ability_score':
            print("len(fused_reason_data_list))", len(fused_reason_data_list))
            for i in tqdm(range(start_index_of_reasoning_ability_score, len(fused_reason_data_list))):
                if new_reason_data_from_VLM[i]['task_type'] == 'default':
                    pass
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_reason = executor.submit(get_reasoning_ability_score, reasoning_ability_checking_agent, new_reason_data_from_VLM[i]['output'])
                        future_description = executor.submit(get_reasoning_ability_score, reasoning_ability_checking_agent, new_description_data_from_VLM[i]['output'])
                        future_outcome = executor.submit(get_reasoning_ability_score, reasoning_ability_checking_agent, new_outcome_data_from_VLM[i]['output'])

                        reasoning_ability_score_for_reason = future_reason.result()
                        reasoning_ability_score_for_description = future_description.result()
                        reasoning_ability_score_for_outcome = future_outcome.result()

                    avg_reasong_ability_score = (reasoning_ability_score_for_reason + reasoning_ability_score_for_description + reasoning_ability_score_for_outcome) / 3
                    sum_reasoning_ability_score = reasoning_ability_score_for_reason + reasoning_ability_score_for_description + reasoning_ability_score_for_outcome
                    score = {
                        'reasoning_ability_score_for_reason': reasoning_ability_score_for_reason,
                        'reasoning_ability_score_for_description': reasoning_ability_score_for_description,
                        'reasoning_ability_score_for_outcome': reasoning_ability_score_for_outcome,
                        'average_reasoning_ability_score': avg_reasong_ability_score,
                        'sum_reasoning_ability_score': sum_reasoning_ability_score,
                        'note': 'each score is from 0 or 1',
                        'video_id': i,
                        'new_reason_data_from_VLM':new_reason_data_from_VLM[i]['output'],
                        'new_description_data_from_VLM':new_description_data_from_VLM[i]['output'],
                        'new_outcome_data_from_VLM':new_outcome_data_from_VLM[i]['output'],
                    }
                    progress_data['reasoning_ability_score'].append(score)
                    
                    progress_data['current_step'] = 'reasoning_ability_score'
                    progress_data['index'] = i
                    save_progress(progress_data, progress_file)
            progress_data['current_step'] = 'finished'
            progress_data['index'] = 9999999999999999999999999
            save_progress(progress_data, progress_file)



    
    print(f"Total Matching Scores: {len(progress_data['matching_score'])}")
    print(f"Total Logic Scores: {len(progress_data['logic_score'])}")
    print(f"Total Key Information Scores: {len(progress_data['key_information_score'])}")
    print(f"Total Reasoning Ability Scores: {len(progress_data['reasoning_ability_score'])}")

    # Save final scores to a JSON file
    scores = {
        'matching_score': progress_data['matching_score'],
        'logic_score': progress_data['logic_score'],
        'key_information_score': progress_data['key_information_score'],
        'reasoning_ability_score': progress_data['reasoning_ability_score']
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, 'score.json')
    with open(file_path, 'w') as f:
        json.dump(scores, f, indent=4)
if __name__ == "__main__":
    main()
