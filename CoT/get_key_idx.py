
import os
import json
import torch

from openai import OpenAI
from tqdm import tqdm
from transformers import MistralForCausalLM, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# ckpt_path = '/data/vllm/ckpt/AI-ModelScope/Mistral-7B-Instruct-v0___2'
# model = MistralForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to('cuda:6')
# tokenizer = AutoTokenizer.from_pretrained(ckpt_path)


PROMPT = """
I have four sentences and one question. I want you to rank the sentences based on their relevance to the question. The higher the relevance, the more likely the sentence is to answer or address the question. Your task is to order the sentences based on relevance from highest to lowest.
The specific requirements are as follows:
You should analyze the semantic relevance between the four sentences and the question. Based on the degree of relevance, you should rank the four sentences, with the most relevant sentence placed first and the least relevant sentence placed last. The output format must be a strict numerical list, indicating the ranking of the sentences from highest to lowest relevance, in the form of [1,4,3,2].You just need to output a list, you don't need anything else.
"""

CASE1 = """
Question:
Give a detailed description of the anomalous segment in the video. Please remember to describe the details of the incident

Sentences:
1.The video shows a man walking down a staircase and being greeted by a group of people. They then proceed to have a conversation.
2.The video shows a woman walking in a park, followed by a group of people playing with fencing swords. The woman then joins in on the game and starts playing with the others.
3.The video shows a man in a blue jacket and a woman in a green hoodie practicing martial arts moves in a parking lot. They are being watched by a group of people who are also practicing martial arts. The man and woman continue to practice while being watched by the group.
4.The video shows a man driving a car and talking to the camera. He then gets out of the car and walks away.
"""
CASE = """
Question:
Give a detailed description of the anomalous segment in the video. Please remember to describe the details of the incident

Sentences:
1.{}
2.{}
3.{}
4.{}
"""

def get_res(res):
    import re
    res = res.split('[/INST]')[-1]
    pattern = r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]'

    # 查找所有匹配项
    matches = re.findall(pattern, res)
    try:
        res = []
        for s in matches[0]:
            if s > '0' and s <= '4':
                res.append(int(s))
        return res
    except:
        return [1,2,3,4,5]


# def get_idx(captions):
#     query = CASE.format(captions[0], captions[1], captions[2], captions[3])
    
#     messages = [
#         {"role": "user", "content":PROMPT + query}
#     ]

#     encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

#     model_inputs = encodeds.to(model.device)
#     generated_ids = model.generate(model_inputs, max_new_tokens=2000, do_sample=False)
#     decoded = tokenizer.batch_decode(generated_ids)
#     return decoded[0]





def get_res_use_gpt(captions):
    query = CASE.format(captions[0], captions[1], captions[2], captions[3])
    client = OpenAI(
        base_url="https://api.xiaoai.plus/v1",
        api_key="sk-DXyiueWuvW1zhqmSBb9792CbD9F9477bB759D81c981fC0A3",
    )
    try:
        # Compute the correctness score
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": PROMPT,
                },
                {
                    "role": "user",
                    "content":query,
                }
            ]
        )
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        return response_message
    except:
        pass


captions = [
        "The video shows a large ship sailing in the ocean when it is hit by another ship, causing it to tilt to one side and eventually sink. The video also shows the aftermath of the collision, with the damaged ship and debris in the water. ",
        "The video shows a large ship sailing in the ocean when it suddenly capsizes and sinks. The ship is seen floating upside down in the water, and a smaller boat is shown nearby. ",
        "The video shows a large ship in the water with two people in yellow and red suits standing on the side. A smaller boat approaches, and a man in a red suit jumps onto the larger ship, while another man in a yellow suit is pulled up by a crane. The video ends with the man in the yellow suit standing on the ship. ",
        "The video shows a large ship in the water being towed by another ship. The water is rough, and the ship being towed is tilted to one side. There is also a smaller boat in the water. "
    ]
res = get_res_use_gpt(captions)
print(res)

def main():
    caption_file = '/data/vllm/VideoLLaMA2-main/CoT/all_captions.json'
    idx_list = []
    idx_dir = '/data/vllm/VideoLLaMA2-main/CoT/key_frame_idx'
    all_idx_file = '/data/vllm/VideoLLaMA2-main/CoT/all_idx.json'

    with open(caption_file, 'r') as fp:
        data = json.load(fp)
        for item in tqdm(data, total=len(data)):
            video = item['video']
            captions = item['captions']
            dic = {}
            dic['video'] = video
            res = get_res_use_gpt(captions)
            res = get_res(res)
            dic['idx'] = res
            print(res)
            if len(res) == 5:
                continue
            with open(os.path.join(idx_dir, video.split('.')[0] + '.json'),'w') as fp:
                json.dump(dic, fp, indent=4)
            idx_list.append(dic)
    with open(all_idx_file, 'w') as fp:
        json.dump(idx_list, fp, indent=4)
                

# if __name__ == "__main__":
#     # main()





