import sys
sys.path.append('/data/vllm/VideoLLaMA2-main')
from PIL import Image


from transformers import CLIPProcessor, CLIPModel
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image

from decord import VideoReader, cpu

def load_video(video_path):
    vr = VideoReader(video_path)
    total_frame_num, fps = len(vr), vr.get_avg_fps()
    frames_to_extract = [int(i*fps) for i in range(int(total_frame_num / fps)) if int(i*fps) < total_frame_num]
    img_array = vr.get_batch(frames_to_extract).asnumpy()
    pil_imgs = [Image.fromarray(img).convert("RGB") for img in img_array]
    return pil_imgs

# model = CLIPModel.from_pretrained("/data/vllm/ckpt/AI-ModelScope/clip-vit-large-patch14").cuda()
# processor = CLIPProcessor.from_pretrained("/data/vllm/ckpt/AI-ModelScope/clip-vit-large-patch14")

# image_path = '/data/OIP-C.jpg'

# image = Image.open(image_path)

# inputs = processor(text=["accident", "two cars on the road"], images=image, return_tensors="pt", padding=True)

# inputs = {k:v.cuda() for k,v in inputs.items()}

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print(probs)


video_path = '/data/vllm/datasets/othervideos/combine_dataset/00016.mp4'
tensor = load_video(video_path)
print(tensor[0])