from config import OPENAI_API_KEY, OPENAI_BASE_URL,MODEL_TYPE,PROMPT_FOR_TEXT_MATCHING,PROMPT_FOR_LOGIC_CHECKING,PROMPT_FOR_KEY_INFOMATION_CHECKING,PROMPTS_FOR_REASONING_ABILITY_CHECKING
from agents.text_matching_agent_by_gpt import TextMatchingAgentByGPT
from agents.logic_checking_agent_by_gpt import LogicCheckingAgent
from agents.key_information_checking_agent import KeyInformationCheckingAgent
from agents.reasoning_ability_checking_agent import ReasoningAbilityCheckingAgent
from utils.data_loader import load_data_from_json_file, load_data_from_xlsx_file
import json
import os
import concurrent.futures

def get_fused_data(base_info_list, base_info_name,data_from_VLM_list,data_from_VLM_name):
    fused_data = []
    for base_info,data_from_VLM in zip(base_info_list,data_from_VLM_list):
        fused_entry = {
            'base infomation': base_info[f'{base_info_name}'],
            'system output': data_from_VLM[f'{data_from_VLM_name}']
        }
        fused_entry_str = str(fused_entry)  
        fused_data.append(fused_entry_str)
    return fused_data

def get_fused_data_with_key_information(base_info_list, base_info_name,data_from_VLM_list,data_from_VLM_name):
    fused_data = []
    for base_info,data_from_VLM in zip(base_info_list,data_from_VLM_list):
        fused_entry = {
            'key infomation': base_info[f'{base_info_name}'],
            'system output': data_from_VLM[f'{data_from_VLM_name}']
        }
        fused_entry_str = str(fused_entry)  
        fused_data.append(fused_entry_str)
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


    #step 1: create scoring agent
    text_matching_agent = TextMatchingAgentByGPT(api_key=OPENAI_API_KEY,base_url=OPENAI_BASE_URL,prompt=PROMPT_FOR_TEXT_MATCHING,model=MODEL_TYPE)
    logic_checking_agent = LogicCheckingAgent(api_key=OPENAI_API_KEY,base_url=OPENAI_BASE_URL,prompt=PROMPT_FOR_LOGIC_CHECKING,model=MODEL_TYPE)
    key_information_checking_agent = KeyInformationCheckingAgent(api_key=OPENAI_API_KEY,base_url=OPENAI_BASE_URL,prompt=PROMPT_FOR_KEY_INFOMATION_CHECKING,model=MODEL_TYPE)
    reasoning_ability_checking_agent = ReasoningAbilityCheckingAgent(api_key=OPENAI_API_KEY,base_url=OPENAI_BASE_URL,prompt=PROMPTS_FOR_REASONING_ABILITY_CHECKING,model=MODEL_TYPE)


    #step 2: import processed data 
    reason_data_from_VLM = load_data_from_json_file(r"./data/model_output/video_llama_reason.json")
    description_data_from_VLM = load_data_from_json_file(r"./data/model_output/video_llama_description.json")
    outcome_data_from_VLM = load_data_from_json_file(r"./data/model_output/video_llama_outcome.json")
    time_segment_data_from_VLM = load_data_from_json_file(r"./data/model_output/video_llama_time_segment.json")
    data_from_human_expert = load_data_from_xlsx_file(r"./data/ground_truth/annotations.xlsx")
    new_reason_data_from_VLM = load_data_from_json_file(r"./data/model_output/videollama2_new_reason.json")
    new_description_data_from_VLM = load_data_from_json_file(r"./data/model_output/videollama2_new_description.json")
    new_outcome_data_from_VLM = load_data_from_json_file(r"./data/model_output/videollama2_new_outcome.json")

    #step 3: get fused data
    fused_reason_data_list = get_fused_data(base_info_list=data_from_human_expert,base_info_name='X2 - reason（导致异常发生的原因）',data_from_VLM_list=reason_data_from_VLM,data_from_VLM_name='output')
    fused_description_data_list = get_fused_data(base_info_list=data_from_human_expert,base_info_name='X3 - description',data_from_VLM_list=description_data_from_VLM,data_from_VLM_name='output')
    fused_outcome_data_list = get_fused_data(base_info_list=data_from_human_expert,base_info_name='X2 -  result （异常事件导致的结果）',data_from_VLM_list=outcome_data_from_VLM,data_from_VLM_name='output')
    fused_reason_data_for_key_information_checking_list = get_fused_data_with_key_information(base_info_list=data_from_human_expert,base_info_name='X4 - key sentences  （尊重事实的前提下，尽可能写一些（4-6个）独立短句（不超过7个单词），中间不要加逗号）',data_from_VLM_list=reason_data_from_VLM,data_from_VLM_name='output')
    fused_description_data_for_key_information_checking_list = get_fused_data_with_key_information(base_info_list=data_from_human_expert,base_info_name='X4 - key sentences  （尊重事实的前提下，尽可能写一些（4-6个）独立短句（不超过7个单词），中间不要加逗号）',data_from_VLM_list=description_data_from_VLM,data_from_VLM_name='output')
    fused_outcome_data_for_key_information_checking_list = get_fused_data_with_key_information(base_info_list=data_from_human_expert,base_info_name='X4 - key sentences  （尊重事实的前提下，尽可能写一些（4-6个）独立短句（不超过7个单词），中间不要加逗号）',data_from_VLM_list=outcome_data_from_VLM,data_from_VLM_name='output')
    #print('fused_reason_data_for_key_information_checking_list:',fused_reason_data_for_key_information_checking_list[0])


    #step 4: scoring
    # text matching score
    matching_score = []
    for i, (reason, description, outcome, human_data) in enumerate(zip(reason_data_from_VLM, description_data_from_VLM, outcome_data_from_VLM, data_from_human_expert)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_reason = executor.submit(get_text_matching_score, text_matching_agent, reason[r'output'], human_data[r'X2 - reason（导致异常发生的原因）'])
            future_description = executor.submit(get_text_matching_score, text_matching_agent, description[r'output'], human_data[r'X3 - description'])
            future_outcome = executor.submit(get_text_matching_score, text_matching_agent, outcome[r'output'], human_data[r'X2 -  result （异常事件导致的结果）'])

            text_matching_score_for_reason = future_reason.result()
            text_matching_score_for_description = future_description.result()
            text_matching_score_for_outcome = future_outcome.result()
        #text_matching_score_for_time_segment = int(text_matching_agent.get_text_matching_score(time_segment[r'output'], human_data[r'X3 - moment']))
        # print(f'matching for the {i}th data')
        # print(f"reason['output']: {reason['output']}")
        # print(f"human_data['X2 - reason（导致异常发生的原因）']: {human_data['X2 - reason（导致异常发生的原因）']}")
        # print(f'text_matching_score_for_reason: {text_matching_score_for_reason}')
        # print(f"description['output']: {description['output']}")
        # print(f"human_data['X3 - description']: {human_data['X3 - description']}")
        # print(f'text_matching_score_for_description: {text_matching_score_for_description}')
        # print(f"outcome['output']: {outcome['output']}")
        # print(f"human_data['X2 -  result （异常事件导致的结果）']: {human_data['X2 -  result （异常事件导致的结果）']}")
        # print(f'text_matching_score_for_outcome: {text_matching_score_for_outcome}')
        # print(f"time_segment['output']: {time_segment['output']}")
        # print(f"human_data['X3 - moment']: {human_data['X3 - moment']}")
        # print(f'text_matching_score_for_time_segment: {text_matching_score_for_time_segment}')
        # print('-----------------------------')
        average_matching_score = (text_matching_score_for_reason + text_matching_score_for_description + text_matching_score_for_outcome ) / 3
        sum_matching_score = text_matching_score_for_reason + text_matching_score_for_description + text_matching_score_for_outcome 
        text_matching_score = {
            'text_matching_score_for_reason': text_matching_score_for_reason,
            'text_matching_score_for_description': text_matching_score_for_description,
            'text_matching_score_for_outcome': text_matching_score_for_outcome,
            'average_matching_score': average_matching_score,
            'sum_matching_score': sum_matching_score,
            'note':'each score is 0 or 1'
        }
        print('text_matching_score:',text_matching_score)
        matching_score.append(text_matching_score)
        # if i == 3:
        #     break

    # logic checking score
    logic_score = []
    for i,(fused_reason_data,fused_description_data,fused_outcome_data) in enumerate(zip(fused_reason_data_list,fused_description_data_list,fused_outcome_data_list)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_reason = executor.submit(get_logic_score, logic_checking_agent, fused_reason_data)
            future_description = executor.submit(get_logic_score, logic_checking_agent, fused_description_data)
            future_outcome = executor.submit(get_logic_score, logic_checking_agent, fused_outcome_data)

            logic_score_for_reason = future_reason.result()
            logic_score_for_description = future_description.result()
            logic_score_for_outcome = future_outcome.result()
        # print('checking for the {}th data'.format(i))
        # print('fused_reason_data:',fused_reason_data)
        # print('logic_score_for_reason:',logic_score_for_reason)
        # print('fused_description_data:',fused_description_data)
        # print('logic_score_for_description:',logic_score_for_description)
        # print('fused_outcome_data:',fused_outcome_data)
        # print('logic_score_for_outcome:',logic_score_for_outcome)
        # print('-----------------------------')
        average_logic_score = (logic_score_for_reason + logic_score_for_description + logic_score_for_outcome) / 3
        sum_logic_score = logic_score_for_reason + logic_score_for_description + logic_score_for_outcome
        score = {
            'logic_score_for_reason': logic_score_for_reason,
            'logic_score_for_description': logic_score_for_description,
            'logic_score_for_outcome': logic_score_for_outcome,
            'average_logic_score': average_logic_score,
            'sum_logic_score': sum_logic_score,
            'note':'each score is from 0 to 5'
        }
        print('score:',score)
        logic_score.append(score)
        # if i == 3:
        #     break

    # key information checking score
    key_information_score = []
    for i, (fused_reason_data_for_key_information_checking,fused_description_data_for_key_information_checking,fused_outcome_data_for_key_informaton_checking) in enumerate(zip(fused_reason_data_for_key_information_checking_list,fused_description_data_for_key_information_checking_list,fused_outcome_data_for_key_information_checking_list)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_reason = executor.submit(get_key_information_checking_score, key_information_checking_agent, fused_reason_data_for_key_information_checking)
            future_description = executor.submit(get_key_information_checking_score, key_information_checking_agent, fused_description_data_for_key_information_checking)
            future_outcome = executor.submit(get_key_information_checking_score, key_information_checking_agent, fused_outcome_data_for_key_informaton_checking)

            key_information_score_for_reason = future_reason.result()
            key_information_score_for_description = future_description.result()
            key_information_score_for_outcome = future_outcome.result()
        # print('checking for the {}th data'.format(i))
        # print('fused_reason_data_for_key_information_checking:',fused_reason_data_for_key_information_checking)
        # print('key_information_score_for_reason:',key_information_score_for_reason)
        # print('fused_description_data_for_key_information_checking:',fused_description_data_for_key_information_checking)
        # print('key_information_score_for_description:',key_information_score_for_description)
        # print('fused_outcome_data_for_key_informaton_checking:',fused_outcome_data_for_key_informaton_checking)
        # print('key_information_score_for_outcome:',key_information_score_for_outcome)
        # print('-----------------------------')
        avg_key_information_score = (key_information_score_for_reason + key_information_score_for_description + key_information_score_for_outcome) / 3
        sum_key_information_score = key_information_score_for_reason + key_information_score_for_description + key_information_score_for_outcome
        score = {
            'key_information_score_for_reason': key_information_score_for_reason,
            'key_information_score_for_description': key_information_score_for_description,
            'key_information_score_for_outcome': key_information_score_for_outcome,
            'average_key_information_score': avg_key_information_score,
            'sum_key_information_score': sum_key_information_score,
            'note':'each score is from 0 to 5'
        }
        key_information_score.append(score)
        print('key_information_score:',score)
        # if i == 3:
        #     break

    # reasoning ability checking score
    reasoning_ability_score = []
    for i, (new_reason_data,new_description_data,new_outcome_data) in enumerate(zip(new_reason_data_from_VLM,new_description_data_from_VLM,new_outcome_data_from_VLM)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_reason = executor.submit(get_reasoning_ability_score, reasoning_ability_checking_agent, new_reason_data['output'])
            future_description = executor.submit(get_reasoning_ability_score, reasoning_ability_checking_agent, new_description_data['output'])
            future_outcome = executor.submit(get_reasoning_ability_score, reasoning_ability_checking_agent, new_outcome_data['output'])

            reasoning_ability_score_for_reason = future_reason.result()
            reasoning_ability_score_for_description = future_description.result()
            reasoning_ability_score_for_outcome = future_outcome.result()
        # print('checking for the {}th data'.format(i))
        # print('new_reason_data:',new_reason_data)
        # print('reasoning_ability_score_for_reason:',reasoning_ability_score_for_reason)
        # print('new_description_data:',new_description_data)
        # print('reasoning_ability_score_for_description:',reasoning_ability_score_for_description)
        # print('new_outcome_data:',new_outcome_data)
        # print('reasoning_ability_score_for_outcome:',reasoning_ability_score_for_outcome)
        # print('-----------------------------')
        avg_reasong_ability_score = (reasoning_ability_score_for_reason + reasoning_ability_score_for_description + reasoning_ability_score_for_outcome) / 3
        sum_reasoning_ability_score = reasoning_ability_score_for_reason + reasoning_ability_score_for_description + reasoning_ability_score_for_outcome
        score = {
            'reasoning_ability_score_for_reason': reasoning_ability_score_for_reason,
            'reasoning_ability_score_for_description': reasoning_ability_score_for_description,
            'reasoning_ability_score_for_outcome': reasoning_ability_score_for_outcome,
            'average_reasoning_ability_score': avg_reasong_ability_score,
            'sum_reasoning_ability_score': sum_reasoning_ability_score,
            'note':'each score is from 0 or 1'
        }
        reasoning_ability_score.append(score)
        print('reasoning_ability_score:',score)
        # if i == 3:
        #     break

    # Save scores to a JSON file
    scores = {
        'matching_score': matching_score,
        'logic_score': logic_score,
        'key_information_score': key_information_score,
        'reasoning_ability_score': reasoning_ability_score
    }
    # Ensure the directory exists
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # Define the file path
    file_path = os.path.join(output_dir, 'scores.json')
    with open(file_path, 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    main()

