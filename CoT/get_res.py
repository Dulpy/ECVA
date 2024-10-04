import os
import sys
import json
import copy
from tqdm import tqdm
import torch
import transformers

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
sys.path.append('/data/vllm/VideoLLaMA2-main')

from peft import PeftModel
from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from videollama2.model.builder import load_pretrained_model


def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

conv_mode = 'mistral'
model_base = '/data/vllm/VideoLLaMA2-main/mistral_ckpt/Mistral-7B-Instruct-v0___2'

model_path = '/data/vllm/VideoLLaMA2-main/work_dirs/mamba_pretrain_v1/mamba_pretrain_v1'

lora_weight = '/data/vllm/VideoLLaMA2-main/work_dirs/mamba_sft/finetune_mamba_sft'
pretrain_mm_mlp_adapter_v2 = '/data/vllm/VideoLLaMA2-main/work_dirs/mamba_sft/finetune_mamba_sft/non_lora_trainables.bin'

model_name = get_model_name_from_path(model_path)


tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)
model = model.to('cuda')


# vision_tower_copy = copy.deepcopy(model.model.vision_tower)


peft_model = PeftModel.from_pretrained(model, lora_weight).cuda()

# peft_model.base_model.model.model.vision_tower = vision_tower_copy

mm_projector_weights = torch.load(pretrain_mm_mlp_adapter_v2, map_location='cpu')


peft_model.base_model.model.model.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


def inference(video,question,chunk_list):
    # Video Inference

    paths = [video]
    questions = [question]
    
    
    if '.jpg' in video or 'png' in video:
        modal_list = ['image']
    else:
        modal_list = ['video']
    # 2. Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor = process_video(paths[0], processor, model.config.image_aspect_ratio, num_frames=32).to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]

    question = default_mm_token + "\n" + questions[0]

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda')

    

    if chunk_list[0] > chunk_list[1]:
        cur_tensor = [tensor[8*(chunk_list[1] - 1):8*chunk_list[1]]] + [tensor[8*(chunk_list[0] - 1):8*chunk_list[0]]]
    else:
        cur_tensor = [tensor[8*(chunk_list[0] - 1):8*chunk_list[0]]] + [tensor[8*(chunk_list[1] - 1):8*chunk_list[1]]]


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=cur_tensor,
            modal_list=modal_list,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs[0]




def main():

    prompts = {
        'Description': """Give a detailed description of the anomalous segment in the video. Please remember to describe the details of the incident.""",

        'Cause': """Please analyze the content of the video and comprehensively summarize the reasons causing the abnormal events. Ensure that you cover all possible reasons for the abnormal events as comprehensively as possible. The output format must strictly follow the following formats:1.xxxx\n2.xxxxx\nFor example,1.The quarrel between the two men became more and more serious.\n2.The car that collided was speeding.\n3.The thief wanted to steal valuables.""",
        
        'Result':  """Please analyze the content of the video and comprehensively summarize the outcomes caused by the abnormal events. Ensure that you cover all abnormal events and their resulting outcomes as comprehensively as possible. The output format must strictly follow the following formats:1.xxxx\n2.xxxxx\nFor example,1.The vehicle was badly damaged.\n2.The window of the car was broken by the impact.\n3.The store's window was shattered.\n4.Several valuable items were stolen.""",
    }

    res_dir = '/data/vllm/VideoLLaMA2-main/CoT/result_sota_cot'
    res_file = '/data/vllm/VideoLLaMA2-main/CoT/all_sota_cot_res.json'
    video_dir = '/data/vllm/datasets/othervideos/combine_dataset'
    idx_file = '/data/vllm/VideoLLaMA2-main/CoT/all_sota_idx.json'
    json_file = "/data/vllm/CUVA/Data/cuva_gt_cleaned.json"
    idx_dic = {}

    with open(idx_file, 'r') as fp:
        tmp = json.load(fp)
        for item in tmp:
            idx_dic[item['video']] = item['idx']
    
    all_res = []
    with open(json_file, 'r') as fp:
        data = json.load(fp)
        for item in tqdm(data[4000:], total=len(data[4000:])):
            try:
                instruction = item['instruction']
                video = item['visual_input']
                id = item['ID']
                task = item['task']
                answer = item['output']

                if task in prompts.keys():
                    instruction = prompts[task]
                else:
                    continue

                response = inference(os.path.join(video_dir, video), instruction, idx_dic[video])
                print("ins:",instruction)
                print("ans:",answer)
                print("pred:",response)
                res = {
                    "instruction":instruction,
                    "video":video,
                    "id":id,
                    "task":task,
                    "answer":answer,
                    "response":response}
                all_res.append(res)
                with open(os.path.join(res_dir, f'{id}.json'),'w') as fp:
                    json.dump(res, fp)
            except:
                continue
    with open(res_file, 'w') as fp:
        json.dump(all_res, fp, indent=4)


if __name__ == "__main__":
    main()




