#    Copyright 2024 Alibaba DAMO Academy
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
from transformers import TRANSFORMERS_CACHE


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'mm_projector.bin')):
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "mamba_scan":
        return MambaScan(config)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)

from ..mamba_arch import MambaModel
from einops import rearrange

class MambaScan(nn.Module):
    def __init__(self, config, drop_path_rate=0.1,mamba_depth=1, mlp_depth=2, num_frames=16, num_patches=256):
        super().__init__()

        self.mambaModel_h = MambaModel(config.mm_hidden_size, drop_path_rate=drop_path_rate, depth=mamba_depth)
        self.readout = build_mlp(mlp_depth, config.mm_hidden_size, config.hidden_size)
    
    def forward(self, video_spatio_temporal_features):
        
        b, f, p, d = video_spatio_temporal_features.shape
        w, h = int(p**0.5), int(p**0.5)
        video_spatio_temporal_features = rearrange(video_spatio_temporal_features, 'b f (h w) d -> b f h w d',
                                                    h=h, w=w)
        video_spatio_temporal_features = rearrange(video_spatio_temporal_features, 'b f h w d -> (b h w) f d', b=b, w=w,
                                                    f=f)
        x = rearrange(video_spatio_temporal_features, '(b h w) f d -> b (f h w) d', b=b, w=w, f=f)
        x = self.mambaModel_h(x)

        pool = torch.nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=0)
        x = rearrange(x, 'b (f h w) d -> b d f h w', b=b, f=f, w=w, d=d)
        video_spatio_temporal_features = rearrange(pool(x), 'b d f h w -> b (f h w) d')
        
        return self.readout(video_spatio_temporal_features)