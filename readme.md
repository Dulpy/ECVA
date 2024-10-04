# CUVA-plus

## Overview

This project is a **multi-modal video large model** designed to handle various video-based tasks using different data modalities like text, audio, and visual inputs. The model supports a wide range of tasks, such as **video classification**, **captioning**, and **action recognition**. It is built to be flexible and scalable for a variety of datasets and applications.

### Key Features:
- **Multi-Modal Processing**: Supports video and text inputs.
- **Flexible Model Design**: Can be easily adapted to different downstream tasks.
- **Complete Pipeline**: Includes training, inference, and evaluation.

---

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Acknowledgement](#Acknowledgement)
- [License](#license)

---

## Installation

To install and set up the environment, follow these steps:

```bash
git clone https://github.com/yourusername/multimodal-video-large-model.git
cd multimodal-video-large-model
pip install -r requirements.txt
```

## Dataset Preparation

You should re-organize the annotated video/image sft data according to the following format and place the image/video data in the path **CUVA-plus/datasets/pretraining/** and **CUVA-plus/datasets/videosft/**

```bash
[
    {
        "id": 0,
        "video": "images/xxx.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat are the colors of the bus in the image?"
            },
            {
                "from": "gpt",
                "value": "The bus in the image is white and red."
            },
            ...
        ],
    }
    {
        "id": 1,
        "video": "videos/xxx.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nWhat are the main activities that take place in the video?"
            },
            {
                "from": "gpt",
                "value": "The main activities that take place in the video are the preparation of camera equipment by a man, a group of men riding a helicopter, and a man sailing a boat through the water."
            },
            ...
        ],
    },
    ...
]
```

## Model Training

1. Prepare CLIP and Mistral Weight
    
 - For Vision-Encoder, similar to most multi-modal large models, CUVA-plus uses the CLIP series as the visual encoder. You can download the related pre-trained weights from [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14).
    
 - For the base model, we utilize the powerful Mistral series to help analyze the video content and provide reliable, accurate answers. You can download the related pre-trained weights from [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).


2. Pretrain Command
```bash
cd CUVA-plus/scripts/vllava/mamba/
./pretrain.sh
```
3. Video SFT Command
```bash
cd CUVA-plus/scripts/vllava/mamba/
./finetune.sh
```

## Inference

Video/Image Inference. We have inherited the inference code from **VideoLLaMA2**.

```python

import os
import sys
import torch
import transformers

sys.path.append('../MLLM')
from conversation import conv_templates
from constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from model.builder import load_pretrained_model

model_path = 'work_dirs/mamba_pretrain/mamba_pretrain_v1'

model_name = get_model_name_from_path(model_path)

model_base = 'Mistral-7B-Instruct-v0___2'

tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)
model = model.to('cuda')

conv_mode = 'mistral'

### Load LoRA weight
from peft import PeftModel
lora_weight = ''
print(f"Loading LoRA weights from {lora_weight}")
peft_model = PeftModel.from_pretrained(model, lora_weight).cuda()

### Load Mamba & MLP weight
pretrain_mm_mlp_adapter_v2 = 'non_lora_trainables.bin'

mm_projector_weights = torch.load(pretrain_mm_mlp_adapter_v2, map_location='cpu')


def get_w(weights, keyword):
    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

peft_model.base_model.model.model.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)

### inference
def inference(question, video):
    # Video Inference
    paths = [video]
    questions = [question]
    
    if '.jpg' in video or 'png' in video:
        modal_list = ['image']
    else:
        modal_list = ['video']
    # Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor = process_video(paths[0], processor, model.config.image_aspect_ratio, num_frames=16).to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]
    tensor = [tensor]

    # Text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda')

    with torch.inference_mode():
        output_ids = peft_model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs[0]

video_path = 'test.mp4'

question = "Please analyze the content of the video and comprehensively summarize the reasons causing the abnormal events. Ensure that you cover all possible reasons for the abnormal events as comprehensively as possible."

res = inference(question,video_path)

print(res)

```


## Evaluation

## Acknowledgement
The codebase of **CUVA-plus** is adapted from [**VideoLLaMA2**](https://github.com/DAMO-NLP-SG/VideoLLaMA2). We are grateful for the foundational work done by the VideoLLaMA2 team, which has significantly contributed to the development of this project.

## License