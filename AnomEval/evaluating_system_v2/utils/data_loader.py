import json
import pandas as pd


def load_data_from_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sorted_data = sorted(data, key=lambda x: x['video_file'])
    return sorted_data


def load_data_from_xlsx_file(file_path):
    r = pd.read_excel(file_path).to_dict(orient='records')
    filtered_r = [item for item in r if isinstance(item['video_id'], int)]
    r = filtered_r
    return r

if __name__ == "__main__":
    data = load_data_from_json_file(r'E:\research_projects\evaluating_system_v2\output\scores.json')
    print(data['reasoning_ability_score'][0][10])
    # load_data_from_xlsx_file('1')