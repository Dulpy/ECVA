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


def get_caption(video):
    # Video Inference
    paths = [video]
    question = """Please describe the video briefly"""
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

    result = []
    ## 将32帧的视频帧分成四段，每个视频段8帧
    for chunk_id in range(4):
        cur_tensor = [tensor[8*chunk_id:8*chunk_id + 8]]

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
        result.append(outputs[0])
    return result


def main():
    res_dir = '/data/vllm/VideoLLaMA2-main/CoT/captions_sota'
    res_file = 'all_sota_captions.json'
    video_dir = '/data/vllm/datasets/othervideos/combine_dataset'
    all_res = []
    for root, dirs, files in os.walk(video_dir):
        for file in tqdm(files[:500], total=len(files[:500])):
            video_path = os.path.join(root, file)
            try:
                res = {}
                captions = get_caption(video_path)
                res['video'] = file
                res['captions'] = captions
                all_res.append(res)
                with open(os.path.join(res_dir, file.split('.')[0] + '.json'), 'w') as fp:
                    json.dump(res, fp, indent=4)
            except:
                continue

    with open(res_file,'w') as fp:
        json.dump(all_res, fp, indent=4)


if __name__ == "__main__":
    main()




