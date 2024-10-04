from clip_encoder import CLIPVisionTower

from dataclasses import dataclass

@dataclass
class MyDataClass:
    mm_vision_select_feature: str = 'cls_patch'
    mm_vision_select_layer: int = -2

vision_tower = '/data/vllm/ckpt/AI-ModelScope/clip-vit-large-patch14'
args = MyDataClass()

model = CLIPVisionTower(args=args, vision_tower=vision_tower)

from decord import VideoReader, cpu
from PIL import Image
def load_video(video_path):
    vr = VideoReader(video_path)
    total_frame_num, fps = len(vr), vr.get_avg_fps()
    frames_to_extract = [int(i*fps) for i in range(int(total_frame_num / fps)) if int(i*fps) < total_frame_num]
    img_array = vr.get_batch(frames_to_extract).asnumpy()
    pil_imgs = [Image.fromarray(img).convert("RGB") for img in img_array]
    return pil_imgs

video_path = '/data/vllm/datasets/othervideos/combine_dataset/00073.mp4'

video_dir = '/data/vllm/datasets/videochat100k/videochatgpt_tune'

feat_dir = '/data/vllm/datasets/videochat100k/clip_l_14_1024'

import os
import h5py
import torch
from tqdm import tqdm
import numpy as np

model = model.cuda().to(torch.bfloat16)


for root, dirs, files in os.walk(video_dir):
    for file in tqdm(files, desc='process', total=len(files)):
        video_path = os.path.join(root, file)
        feat_path = os.path.join(feat_dir, file.split('.')[0])

        if os.path.exists(f'{feat_path}.h5'):
            continue

        try:
            pil_imgs = load_video(video_path)
            video = model.image_processor(pil_imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16, device='cuda')
            video_feat = model(video)
            video_feat = video_feat.to(torch.float32)
            with h5py.File(f'{feat_path}.h5', 'w') as h5f:
                tensor_np = video_feat.cpu().numpy()
                h5f.create_dataset('tensor_data', data=tensor_np)

        except:
            continue
