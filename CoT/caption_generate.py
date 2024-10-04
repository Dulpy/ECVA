import torch
import transformers
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
sys.path.append('/data/vllm/VideoLLaMA2-main')
from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from videollama2.model.builder import load_pretrained_model

model_path = '/data/vllm/VideoLLaMA2-main/work_dirs/mamba_pretrain_v1/mamba_pretrain_v1'
# Base model inference (only need to replace model_path)
# model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-Base'
model_name = get_model_name_from_path(model_path)
# model_name = 'DAMO-NLP-SG/VideoLLaMA2-7B'
model_base = '/data/vllm/VideoLLaMA2-main/mistral_ckpt/Mistral-7B-Instruct-v0___2'
print(model_name)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)
model = model.to('cuda')

conv_mode = 'mistral'


#### =========================加载lora权重======================== ####
from peft import PeftModel
lora_weight = '/data/vllm/VideoLLaMA2-main/work_dirs/mamba_sft/finetune_mamba_sft_827_2stage_1mamba_2mlp'
print(f"Loading LoRA weights from {lora_weight}")
model = PeftModel.from_pretrained(model, lora_weight).cuda()



#### =========================加载mm-connecter权重======================== ####
pretrain_mm_mlp_adapter_v2 = '/data/vllm/VideoLLaMA2-main/work_dirs/mamba_sft/finetune_mamba_sft_827_2stage_1mamba_2mlp/non_lora_trainables.bin'
# CC3M-575 + ShareGPT4V + ShareGPT4o版本

mm_projector_weights = torch.load(pretrain_mm_mlp_adapter_v2, map_location='cpu')


def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

model.base_model.model.model.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)



def inference(question, video):
    # Video Inference
    paths = [video]
    questions = [question]
    # Reply:
    # The video features a kitten and a baby chick playing together. The kitten is seen laying on the floor while the baby chick hops around. The two animals interact playfully with each other, and the video has a cute and heartwarming feel to it.
    if '.jpg' in video or 'png' in video:
        modal_list = ['image']
    else:
        modal_list = ['video']
    # 2. Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor = process_video(paths[0], processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
    tensor = [tensor]

    # 3. text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda')

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=False,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs[0]

#### ========================= 其他代码逻辑 ============================== ####
video_path = '/data/vllm/datasets/processed_combine_dataset/00003.mp4'
# /data/vllm/datasets/othervideos/combine_dataset
# /data/vllm/datasets/processed_combine_dataset
question = """Please analyze the content of the video and comprehensively summarize the reasons causing the abnormal events. Ensure that you cover all possible reasons for the abnormal events as comprehensively as possible."""
res = inference(question,video_path)
print(res)