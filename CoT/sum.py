import json
import os
all_captions = []

for root,dirs,files in os.walk('/data/vllm/VideoLLaMA2-main/CoT/captions_sota'):
    for file in files:
        with open(os.path.join(root,file),'r') as fp:
            data = json.load(fp)
            dic = {}
            dic['video'] = data['video']
            dic['captions'] = data['captions']
            all_captions.append(dic)

with open('all_sota_captions.json','w') as fp:
    json.dump(all_captions, fp, indent=4)